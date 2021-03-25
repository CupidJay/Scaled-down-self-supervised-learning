python main_caltech101.py \
  -a resnet50 \
  --lr 10.0 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --linear-eval \
  --gpus 4,5,6,7 \
  --num-classes 102 \
  --save-dir caltech_checkpoints \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_800ep_pretrain.pth.tar \
  /opt/caoyh/datasets/caltech101/v1