python main.py \
  -a resnet50 \
  --lr 0.001 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_smallIN_class_1000_n_10_8000ep_pretrain_112x112.pth.tar \
  --gpus 12,13,14,15 \
  --num-classes 100 \
  /opt/caoyh/datasets/aircraft