python main.py \
  -a resnet50 \
  --lr 10.0 \
  --linear-eval \
  --dataset-name cifar10 --num-classes 10 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained resnet50 \
  --gpus 4,5,6,7 \
  /opt/caoyh/datasets/CIFAR10