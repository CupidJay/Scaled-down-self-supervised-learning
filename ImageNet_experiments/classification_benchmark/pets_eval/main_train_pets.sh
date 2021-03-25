python main.py \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 64 --epochs 120 \
  --step-lr --schedule 40 80 \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_800ep_pretrain.pth.tar \
  --random-resized-crop \
  --gpus 12,13,14,15 \
  --num-classes 37 \
  /opt/caoyh/datasets/pets