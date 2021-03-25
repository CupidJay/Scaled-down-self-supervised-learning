We follow [official MoCo training code](https://github.com/facebookresearch/moco) and please refer to the original re for more details.

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


### Pre-Training on ImageNet

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine (c.f. moco_imagenet.sh), run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 --epochs 200 \
  --input-size [112 for small resolution or 224 for baseline] \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mlp --moco-t 0.2 --aug-plus --cose \
  [your imagenet-folder with train and val folders]
```
### Pre-Training on small ImageNet

- First we create small ImageNet, run:

```
python create_small_imagenet.py \
```
- For unsupervised pre-training, we run moco_imagenet.sh as above and just change [imagenet-folder] to [small-imagenet-folder]
- For supervised pre-training, we run supervised_training.sh:
```
python main_supervised.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 256 --epochs 800 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cos --eval-freq 100 \
  [your small-imagenet-folder with train and val folders]
```

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine (c.f. lincls.sh), run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

### Transferring to Object Detection

See [./detection](detection).

### Transferring to other classification benchmarks

See [./classfication_benchmark](classfication_benchmark).