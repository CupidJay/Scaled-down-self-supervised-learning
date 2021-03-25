import os

import numpy as np
from PIL import Image
import torch
import torchvision

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

voc_class_list = ['aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']


class VOC2007(torchvision.datasets.VOCDetection):
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        super().__init__(root=root, year='2007', image_set=image_set, download=True)
        self.class_list = voc_class_list
        self.class2id = {self.class_list[i]: i for i in range(len(self.class_list))}
        self.targets = self._init_targets()
        self.transform = transform
        self.target_transform = target_transform
        self.label_dir = os.path.join(root, 'VOCdevkit/VOC2007/ImageSets/Main')
        self.image_set = image_set

    def _init_targets(self):
        targets = torch.zeros((len(self), 20))
        targets[:] = -1
        for index in range(len(self)):
            img, anno = super(VOC2007, self).__getitem__(index)
            target = anno['annotation']['object']
            for obj in target:
                class_name = obj['name']
                if obj['difficult'] == '0':
                    targets[index, self.class2id[class_name]] = 1
                else:
                    # if raw_label = 1, then set 1
                    # elif raw_label <= 0, then set 0
                    raw_label = targets[index, self.class2id[class_name]]
                    targets[index, self.class2id[class_name]] = max(0, raw_label)
        return targets

    def __getitem__(self, index):
        img, anno = super(VOC2007, self).__getitem__(index)
        target = self.targets[index]
        name = os.path.basename(self.images[index]).split('.')[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index, name

    def eval(self, predict, targets):
        ap = np.zeros(20)
        for i in range(20):
            ap[i] = VOC2012AP(predict[:, i], targets[:, i])
        return ap

    def eval_file(self, results_dir):
        ap = np.zeros(20)
        for i in range(20):
            cls_name = self.class_list[i]
            filename = '{}_{}.txt'.format(cls_name, self.image_set)
            predict_file = os.path.join(results_dir, filename)
            label_file = os.path.join(self.label_dir, filename)
            label = dict()
            with open(label_file, 'r') as f:
                results = f.readlines()
            for res in results:
                res = res.rstrip()
                name, value = res.split()
                label[name] = float(value)

            with open(predict_file, 'r') as f:
                predict = f.readlines()

            predict_np = np.zeros(len(predict))
            gt_np = np.zeros(len(predict))
            for j, pre in enumerate(predict):
                pre = pre.rstrip()
                name, value = pre.split()
                predict_np[j] = float(value)
                gt_np[j] = label.get(name, 0)
            ap[i] = VOC2012AP(predict_np, gt_np)
        return ap

    def show_AP(self, AP, print_func=print):
        '''
        voc_class = [' aero ', ' bike ', ' bird ', ' boat ', 'bottle', ' bus  ', ' car  ', ' cat  ',
                    'chair ', ' cow  ', 'table ', ' dog  ', 'horse ', 'mbike ', 'person', 'plant ',
                    'sheep ', ' sofa ', 'train ', '  tv  ']
                    '''
        class_str = ' aero | bike | bird | boat |bottle| bus  | car  | cat  |chair | cow  '
        class_str += '|table | dog  |horse |mbike |person|plant |sheep | sofa |train |  tv  '

        AP_str = ['{:4.1f}'.format(ap * 100) for ap in AP]
        print_func("Cls{}".format(class_str))
        print_func("AP: {}".format(' & '.join(AP_str)))


def VOCap(rec, prec):
    length = len(rec)
    mrec = np.zeros(length + 2)
    mrec[-1] = 1
    mpre = np.zeros(length + 2)
    for i in range(length, 0, -1):
        mpre[i] = max(prec[i - 1], mpre[i + 1])
        mrec[i] = rec[i - 1]
    mpre[0] = max(0, mpre[1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[i] - mrec[i - 1]) * mpre[i]).sum()
    return ap


def VOC2012AP(predict_np, gt_np):
    # gt_np: length #imgs
    # gt_np: 1 positive, -1 negtive, 0 difficult
    indices = predict_np.argsort()[::-1]
    tp = gt_np[indices] > 0
    fp = gt_np[indices] < 0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (gt_np > 0).sum()
    prec = tp / (fp + tp + 1e-8)

    ap = VOCap(rec, prec)

    return ap


def check_target(path, image_set='train'):
    dataset = VOC2007(path, image_set)
    label_dir = dataset.label_dir
    label = dict()
    for c in voc_class_list:
        label[c] = dict()
        filename = '{}_{}.txt'.format(c, image_set)
        label_file = os.path.join(label_dir, filename)
        with open(label_file, 'r') as f:
            results = f.readlines()
        for res in results:
            res = res.rstrip()
            name, value = res.split()
            label[c][name] = float(value)

    for i, data in enumerate(dataset):
        img, target, index, name = data
        for j in range(20):
            if target[j] != label[voc_class_list[j]][name]:
                import IPython
                IPython.embed()
    print('no error')
