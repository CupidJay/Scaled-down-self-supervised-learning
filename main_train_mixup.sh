python main.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 120 \
  --gpus 12,13,14,15 \
  --mixup --alpha 1.0 \
  --pretrained /opt/caoyh/code/SSL/random_potential/checkpoints/moco_finetune/scda_r50_moco_cub_200ep_112x112_pretrain.pth.tar \
  --num-classes 200 \
  /opt/caoyh/datasets/cub200