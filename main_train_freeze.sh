python main_freeze.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 10 \
  --step-lr --freeze \
  --gpus 8,9,10,11 \
  --save-dir cub_checkpoints \
  --pretrained [path to your pretrained model] \
  --num-classes 200 \
  /opt/caoyh/datasets/cub200