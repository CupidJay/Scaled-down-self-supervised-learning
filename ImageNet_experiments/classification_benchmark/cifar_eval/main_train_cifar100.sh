python main.py \
  -a resnet50 \
  --lr 0.001 \
  --dataset-name cifar100 --num-classes 100 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_200ep_pretrain_112x112.pth.tar \
  --gpus 0,1,2,3 \
  /opt/caoyh/datasets/CIFAR100