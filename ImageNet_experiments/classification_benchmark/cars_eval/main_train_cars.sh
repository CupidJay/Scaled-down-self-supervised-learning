python main.py \
  -a resnet50 \
  --lr 0.001 \
  --batch-size 64 --epochs 120 \
  --step-lr --schedule 40 80 \
  --random-resized-crop \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_200ep_pretrain_112x112_finetune_50ep_lr_0.03_224x224.pth.tar \
  --gpus 0,1,2,3 \
  --num-classes 196 \
  /opt/caoyh/datasets/cars