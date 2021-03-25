import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from voc_dataset import VOC2007
from eval import validate_voc, validate_voc_file, validate_coco, prec_recall_for_batch
from utils import save_checkpoint, AverageMeterSet, accuracy, parameters_string, load_pretrained_model


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

input_size = 224
batch_size = 16
eval_batch_size = 64

thre=0.5
print_freq = 50

#optimization
weight_decay = 1e-5
momentum = 0.9
nesterov = False
epochs = 30
lr_reduce_epochs = '10, 20'

# set False for full finetune
finetune_fc = False
lr = 0.01

# set True for linear evaluation
#finetune_fc = True
#lr = 10.0

arch = 'resnet18'

#pretrained = arch
pretrained = '../simclr/voc_checkpoints/simclr_voc_800ep_lr_0.5_bs_512_112x112_pretrain.pth.tar'
#pretrained = '../checkpoints/moco_v2_200ep_pretrain.pth.tar'
#pretrained = ''
#pretrained = '/opt/caoyh/code/SSL/where_is_the_patch/checkpoints/v2_checkpoint_0000.pth.tar'

evaluate = False
evaluation_epochs = 1
checkpoint_epochs = 80
checkpoint_path = 'checkpoints/VOC2007_'+pretrained.split('/')[-1]+'_'+str(input_size)

if finetune_fc:
    checkpoint_path += 'linear'
else:
    checkpoint_path += 'full'

print('input size', input_size)

print(checkpoint_path)

def voc2007():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    data = '/opt/Dataset'

    train_transformation = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transformation = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    def target_transform(target):
        # use difficult label
        return (target >= 0).float()

    train_dataset = VOC2007(data, 'trainval', transform=train_transformation, target_transform=target_transform)
    val_dataset = VOC2007(data, 'test', transform=eval_transformation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': 20
    }



best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1

    start_time = time.time()
    dataset_config = voc2007()
    train_dataset = dataset_config.get('train_dataset')
    val_dataset = dataset_config.get('val_dataset')
    num_classes = dataset_config.get('num_classes')
    #train_loader = create_train_loader(train_dataset, args=args)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)

    if val_dataset is not None:
        eval_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False)
    else:
        eval_loader = None
    print("=> load dataset in {} seconds".format(time.time() - start_time))

    #if 'voc' in args.dataset:
    validate = validate_voc
    #else:
        #validate = validate_coco

    print("=> creating model ")

    model = models.__dict__[arch](num_classes=num_classes)


    #print(parameters_string(model))

    model = load_pretrained_model(model, pretrained)
    model = model.cuda()
    model = nn.DataParallel(model)

    class_criterion = nn.BCEWithLogitsLoss()

    if finetune_fc:
        print('=> Finetune only FC layer')
        paras = model.module.fc.parameters()

        #print('=> Finetune only FC + layer4')
        #paras = [{'params': model.module.fc.parameters(),
        #          'params': model.module.layer4.parameters()}]
    else:
        print('=> Training all layers')
        paras = model.parameters()

    print('start learning rate ', lr)
    optimizer = torch.optim.SGD(paras, lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    if evaluate:
        results_dir = './predict'
        validate_voc_file(eval_loader, model, thre, 0, print_freq, results_dir)
        #validate(eval_loader, model, args.thre, 0, context.vis_log, LOG, args.print_freq)

    for epoch in range(epochs):

        # train for one epoch
        train(train_loader, model, class_criterion, optimizer, epoch)

        if evaluation_epochs and (epoch + 1) % evaluation_epochs == 0 and eval_loader is not None:

            prec1 = validate(eval_loader, model, thre, epoch, print_freq)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        if checkpoint_epochs and (epoch + 1) % checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

    save_checkpoint({
        'epoch': epoch + 1,
        'global_step': global_step,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, False, checkpoint_path, 'final')
    print("best_prec1 {}".format(best_prec1))


def train(train_loader, model, class_criterion, optimizer, epoch):
    global global_step
    start_time = time.time()

    Sig = torch.nn.Sigmoid()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data[0], data[1]
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        input, target = input.cuda(), target.float().cuda()

        model_out = model(input)
        if isinstance(model_out, tuple):
            feat, class_logit = model_out
        else:
            class_logit = model_out

        output = Sig(class_logit)
        # output = class_logit
        class_loss = class_criterion(class_logit, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        class_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        global_step += 1

        meters.update('lr', optimizer.param_groups[0]['lr'])
        minibatch_size = len(target)
        meters.update('class_loss', class_loss.item())
        # measure accuracy and record loss
        this_prec, this_rec = prec_recall_for_batch(output.data, target, thre)
        meters.update('prec', float(this_prec), input.size(0))
        meters.update('rec', float(this_rec), input.size(0))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
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


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    new_lr = lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    if lr_reduce_epochs:
        reduce_epochs = [int(x) for x in lr_reduce_epochs.split(',')]
        for ep in reduce_epochs:
            if epoch >= ep:
                new_lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


if __name__ == '__main__':
    main()
