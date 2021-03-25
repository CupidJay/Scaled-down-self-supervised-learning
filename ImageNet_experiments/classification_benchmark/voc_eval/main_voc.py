import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import ramps
from utils import load_pretrained_model, AverageMeterSet

from voc_dataset import VOC2007
from eval import validate_voc, validate_voc_file, validate_coco, prec_recall_for_batch

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--num-classes', default=20, type=int, metavar='N',
                    help='number of classes in the dataset')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--thre', default=0.5, type=int,
                    metavar='N', help='threshold (default: 0.5)')
parser.add_argument('--input-size', default=224, type=int, help='input resolution (default 224x224)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none) used for finetune')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', default='0', type=str)

parser.add_argument('--step-lr', action='store_true', default=False, help='use step LR')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')

parser.add_argument('--linear-eval', action='store_true', default=False, help='linear evaluation on fixed representations')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='where to save models')
#mixup options
parser.add_argument('--mixup', action='store_true', default=False, help='use mixup training')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha value in mixup')

best_acc1 = 0


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.linear_eval:
        args.save_dir = os.path.join(args.save_dir, 'linear_eval')
    else:
        args.save_dir = os.path.join(args.save_dir, 'finetune_all', 'mixup' if args.mixup else 'plain')

    args.save_dir = os.path.join(args.save_dir, 'gpus_{}_lr_{}_bs_{}_epochs_{}_pretrained_{}'.format(len(args.gpus.split(',')),
                                                                                args.lr,
                                                                                args.batch_size,
                                                                                args.epochs,
                                                                                args.pretrained.split('/')[-1]))

    if args.mixup:
        args.save_dir = args.save_dir + '_alpha_{}'.format(args.alpha)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.linear_eval:
        print("=> linear evaluation")
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    else:
        print("=> finetuning all model")
        if args.mixup:
            print("=> using mixup strategy")
        else:
            print("=> plain training")

    # optionally use pretrained weights
    if args.pretrained:
        model = load_pretrained_model(model, args.pretrained)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    if args.linear_eval:
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.input_size == 112:
        print('112x112 input')
        # for 224 input
        train_transforms = transforms.Compose([
            transforms.Resize(size=128),
            # transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=112),
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(size=128),
            transforms.CenterCrop(size=112),
            transforms.ToTensor(),
            normalize
        ])
    elif args.input_size == 224:
        print('224x224 input')
        #for 224 input
        train_transforms = transforms.Compose([
            transforms.Resize(size=256),
            #transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224),
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ])

    elif args.input_size == 448:
        print('448x448 input')
        #for 448 input
        train_transforms = transforms.Compose([
            transforms.Resize(size=448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=448),
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(size=448),
            transforms.CenterCrop(size=448),
            transforms.ToTensor(),
            normalize
        ])

    def target_transform(target):
        # use difficult label
        return (target >= 0).float()

    train_dataset = VOC2007(args.data, 'trainval', transform=train_transforms, target_transform=target_transform)
    val_dataset = VOC2007(args.data, 'test', transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate = validate_voc

    if args.evaluate:
        results_dir = './predict'
        validate_voc_file(val_loader, model, args.thre, 0, args.print_freq, results_dir)
        return

    train_start = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.mixup:
            mixup_train(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, args.thre, epoch, args.print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'acc1': acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_dir)

    print('best acc1', best_acc1)
    train_end = time.time()

    print('total training time elapses {} hours'.format((train_end-train_start)/3600.0))

def train(train_loader, model, criterion, optimizer, epoch, args):
    start_time = time.time()
    meters = AverageMeterSet()

    Sig = torch.nn.Sigmoid()

    # switch to train mode
    """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
    """
    if args.linear_eval:
        model.eval()
    # switch to train mode
    else:
        #model.train()
        model.eval()
    #model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        images, target = data[0], data[1]
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        # measure data loading time
        #data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.float().cuda(args.gpu, non_blocking=True)

        # compute output
        model_output = model(images)
        if isinstance(model_output, tuple):
            feat, class_logit = model_output
        else:
            class_logit = model_output

        output = Sig(class_logit)

        loss = criterion(class_logit, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update('lr', optimizer.param_groups[0]['lr'])
        meters.update('class_loss', loss.item())
        # measure accuracy and record loss
        this_prec, this_rec = prec_recall_for_batch(output.data, target, args.thre)
        meters.update('prec', float(this_prec), images.size(0))
        meters.update('rec', float(this_rec), images.size(0))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec {meters[prec]:.3f}\t'
                'Rec {meters[rec]:.3f}\t'.format(
                    epoch, i, len(train_loader), meters=meters))

    print(' * TRAIN Prec {:.3f} ({:.1f}/{:.1f}) Recall {:.3f} ({:.1f}/{:.1f})'
          .format(meters['prec'].avg, meters['prec'].sum / 100, meters['prec'].count,
                  meters['rec'].avg, meters['rec'].sum / 100, meters['rec'].count))
    print("--- training epoch in {} seconds ---".format(time.time() - start_time))

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


#def adjust_learning_rate(optimizer, epoch, args):
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = args.lr * (0.1 ** (epoch // 30))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    #lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    #"""
    if not args.step_lr:
        lr *= ramps.cosine_rampdown(epoch, args.epochs)
    #"""
    #MultiStep LR
    else:
        #if epoch >= 150:
        #    lr /= 10
        #if epoch >= 250:
        #    lr /= 10
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    #"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()