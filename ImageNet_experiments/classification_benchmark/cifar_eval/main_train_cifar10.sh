python main.py \
  -a resnet50 \
  --lr 0.001 \
  --dataset-name cifar10 --num-classes 10 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_custom_res50_200ep_pretrain_112x112.pth.tar \
  --gpus 12,13,14,15 \
  /opt/caoyh/datasets/CIFAR10