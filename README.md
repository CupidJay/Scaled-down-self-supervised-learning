# Scaled-down self-supervised learning
official pytorch implementation of Rethining Self-supervised Learning: Small is Beautiful.

## Abstract
Self-supervised learning (SSL), in particular contrastive
learning, has made great progress in recent years. However,
a common theme in these methods is that they inherit the
learning paradigm from the supervised deep learning scenario.
Current SSL methods are often pretrained for many
epochs on large-scale datasets using high resolution images,
which brings heavy computational cost and lacks flexibility.
In this paper, we demonstrate that the learning paradigm
for SSL should be different from supervised learning and
the information encoded by the contrastive loss is expected
to be much less than that encoded in the labels in supervised
learning via the cross entropy loss. Hence, we propose
scaled-down self-supervised learning (S3L), which include 3
parts: small resolution, small architecture and small data.
On a diverse set of datasets, SSL methods and backbone
architectures, S3L achieves higher accuracy consistently
with much less training cost when compared to previous SSL
learning paradigm. Furthermore, we show that even without
a large pretraining dataset, S3L can achieve impressive
results on small data alone.

## Getting Started

### Prerequisites
* python 3
* PyTorch (>= 1.6)
* torchvision (>= 0.7)
* Numpy

### Small resolution
- Pretraining stage: we use mocov2 for example
```
cd moco
bash finetune_cub.sh
```
- Fine-tuning stage: we use mixup for example
```
bash main_train_mixup.sh
```

## Citation
```
@article{S3L,
   title         = {Rethinking Self-supervised Learning: Small is Beautiful},
   author        = {Yun-Hao Cao and Jianxin Wu},
   year          = {2021},
   journal = {To appear}}
```