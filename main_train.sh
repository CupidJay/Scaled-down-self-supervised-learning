python main.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 120 \
  --gpus 12,13,14,15 \
  --save-dir cub_checkpoints \
  /opt/caoyh/datasets/cub200