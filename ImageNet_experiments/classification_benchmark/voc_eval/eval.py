import time
import os
import torch
import torch.nn as nn

import numpy as np

from utils import AverageMeterSet, accuracy

def validate(eval_loader, model, epoch, print_freq, type_string=''):
    start_time = time.time()
    class_criterion = nn.CrossEntropyLoss().cuda()

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(eval_loader):
        input, target = data[0], data[1]
        meters.update('data_time', time.time() - end)

        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()

            # compute output
            model_out = model(input)
            if isinstance(model_out, tuple):
                feat, class_logit = model_out
            else:
                class_logit = model_out

            class_loss = class_criterion(class_logit, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(class_logit.data, target.data, topk=(1, 5))
        minibatch_size = len(target)
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1, minibatch_size)
        meters.update('top5', prec5, minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'
                .format(i, len(eval_loader), meters=meters))

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    print("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return meters['top1'].avg

def prec_recall_for_batch(output, target, thre):
    pred = output.gt(thre).long()
    this_tp = (pred + target).eq(2).sum()
    this_fp = (pred - target).eq(1).sum()
    this_fn = (pred - target).eq(-1).sum()
    this_tn = (pred + target).eq(0).sum()

    this_prec = this_tp.float() / (
        this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
    this_rec = this_tp.float() / (
        this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0
    return this_prec, this_rec

def average_precision(output, target):
    epsilon = 1e-8
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def coco_metric(scores_, targets_, thre=0.5):
    n, n_class = scores_.shape
    Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        Ng[k] = np.sum(targets == 1)
        Np[k] = np.sum(scores >= thre)
        Nc[k] = np.sum(targets * (scores >= thre))
    Np[Np == 0] = 1
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)
    OP, OR, OF1 = 100 * OP, 100 * OR, 100 * OF1
    CP, CR, CF1 = 100 * CP, 100 * CR, 100 * CF1
    return OP, OR, OF1, CP, CR, CF1

def validate_coco(eval_loader, model, thre, epoch, print_freq):
    start_time = time.time()
    meters = AverageMeterSet()

    model.eval()
    Sig = torch.nn.Sigmoid()

    end = time.time()
    preds = []
    targets = []
    for i, data in enumerate(eval_loader):
        input, target = data[0], data[1]
        meters.update('data_time', time.time() - end)
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        this_prec, this_rec = prec_recall_for_batch(output.data, target, thre)
        meters.update('prec', float(this_prec), input.size(0))
        meters.update('rec', float(this_rec), input.size(0))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {meters[batch_time]:.3f}\t'
                  'Data {meters[data_time]:.3f}\t'
                  'Prec {meters[prec]:.2f}\t'
                  'Recall {meters[rec]:.2f}'
                  .format(i, len(eval_loader), meters=meters))

    targs = torch.cat(targets).numpy()
    preds = torch.cat(preds).numpy()
    OP, OR, OF1, CP, CR, CF1 = coco_metric(preds, targs, thre)
    print(' * CP {:.2f} CR {:.2f} CF1 {:.2f} OP {:.2f} OR {:.2f} OF1 {:.2f}'
          .format(CP, CR, CF1, OP, OR, OF1))

    mAP_score = mAP(targs, preds)
    print(" * TEST [{}] mAP: {}".format(epoch, mAP_score))
    print("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return mAP_score


def validate_voc(eval_loader, model, thre, epoch, print_freq):
    start_time = time.time()

    meters = AverageMeterSet()

    model.eval()
    Sig = torch.nn.Sigmoid()

    end = time.time()
    preds = []
    targets = []
    for i, data in enumerate(eval_loader):
        input, target = data[0], data[1]
        meters.update('data_time', time.time() - end)
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        this_prec, this_rec = prec_recall_for_batch(output.data, target, thre)
        meters.update('prec', float(this_prec), input.size(0))
        meters.update('rec', float(this_rec), input.size(0))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {meters[batch_time]:.3f}\t'
                  'Data {meters[data_time]:.3f}\t'
                  'Prec {meters[prec]:.2f}\t'
                  'Recall {meters[rec]:.2f}'
                  .format(i, len(eval_loader), meters=meters))

    targs = torch.cat(targets).numpy()
    preds = torch.cat(preds).numpy()
    AP = eval_loader.dataset.eval(preds, targs)
    eval_loader.dataset.show_AP(AP)

    mAP = 100 * AP.mean()
    print(" * TEST [{}] mAP: {}".format(epoch, mAP))

    print("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return mAP


def validate_voc_file(eval_loader, model, thre, epoch, print_freq, results_dir):
    start_time = time.time()

    meters = AverageMeterSet()

    model.eval()
    Sig = torch.nn.Sigmoid()

    end = time.time()
    preds = []
    targets = []
    names = []
    for i, data in enumerate(eval_loader):
        assert len(data) >= 4
        input, target, name = data[0], data[1], data[3]
        meters.update('data_time', time.time() - end)
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda()))

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())
        names.extend(name)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {meters[batch_time]:.3f}\t'
                  'Data {meters[data_time]:.3f}\t'
                  .format(i, len(eval_loader), meters=meters))

    preds = torch.cat(preds).numpy()
    targs = torch.cat(targets).numpy()
    if results_dir is not None:
        # save to results dir
        os.makedirs(results_dir, exist_ok=True)
        for i in range(20):
            cls_name = eval_loader.dataset.class_list[i]
            filename = '{}_{}.txt'.format(cls_name, eval_loader.dataset.image_set)
            with open(os.path.join(results_dir, filename), 'w') as f:
                for j in range(len(names)):
                    f.write('{} {}\n'.format(names[j], preds[j, i]))


    AP = eval_loader.dataset.eval_file(results_dir)
    eval_loader.dataset.show_AP(AP, print_func=LOG.info)

    mAP = 100 * AP.mean()
    print(" * TEST [{}] VOC2012 mAP: {}".format(epoch, mAP))

    print("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return mAP
