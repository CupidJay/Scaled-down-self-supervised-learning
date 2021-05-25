# Scaled-down self-supervised learning
official pytorch implementation of Rethinking Self-supervised Learning: Small is Beautiful.

paper is availabel at [[arxiv]](https://arxiv.org/abs/2103.13559)

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
* PyTorch (= 1.6)
* torchvision (= 0.7)
* Numpy
* CUDA 10.1

### Small resolution
- Pre-training stage: we use mocov2 for example (c.f. moco/pretrain_cub.sh), run:
```
cd moco
python main_moco_pretraining.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size [112 or 56 for small resolutions and 224 for baseline] \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --save-dir cub_checkpoints \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  [path to cub200 dataset]

```
  Multi-stage-pre-training: we use 112->224 as an example, first pre-train a model under 112x112 resolution as before, then run:
```
python main_moco_pretraining.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --save-dir cub_checkpoints \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  --pretrained [112 resolution pretrained model]
  [path to cub200 dataset]

```


- Fine-tuning stage: we use mixup for example (c.f. main_train_mixup.sh):
```
python main.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 120 \
  --gpus 12,13,14,15 \
  --mixup --alpha 1.0 \
  --pretrained [path to SSL pre-trained model] \
  --num-classes 200 \
  [path to cub200 dataset]
```

### Small architecture

- Pretraining stage: change the architecture from resnet50 to custom_resnet50 (we remove conv5 in custom_resnet50) in moco/pretrain_cub.sh
- Warm-up conv5 stage: warm up conv5 in custom_resnet50 (c.f. main_train_freeze.sh), run:
```
python main_freeze.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 10 \
  --step-lr --freeze \
  --gpus 8,9,10,11 \
  --save-dir cub_checkpoints \
  --pretrained [path to SSL pretrained custom_resnet50 model] \
  --num-classes 200 \
  [path to cub200 dataset]
```
- Fine-tuning stage, set --pretrained to the model obtained in the previous stage in main_train_mixup.sh

### ImageNet experiments
For ImageNet and small ImageNet experiments, see [./ImageNet_experiments](ImageNet_experiments). The performance of transferring to object detection and other classification benchmarks can also be found here.


## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{S3L,
   title         = {Rethinking Self-supervised Learning: Small is Beautiful},
   author        = {Yun-Hao Cao and Jianxin Wu},
   year          = {2021},
   journal = {arXiv preprint arXiv:2103.13559}}
```
