python main.py \
  -a resnet50 \
  --lr 10.0 \
  --linear-eval \
  --dataset-name cifar100 --num-classes 100 \
  --batch-size 64 --epochs 30 \
  --step-lr --schedule 10 20 \
  --pretrained resnet50 \
  --gpus 0,1,2,3 \
  /opt/caoyh/datasets/CIFAR100