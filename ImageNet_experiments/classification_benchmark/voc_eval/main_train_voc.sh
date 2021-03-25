python main_voc.py \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 64 --epochs 60 \
  --schedule 20 40 \
  --gpus 8,9,10,11 \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_smallIN_class_1000_n_10_8000ep_pretrain_112x112.pth.tar \
  --num-classes 20 \
  /opt/Dataset