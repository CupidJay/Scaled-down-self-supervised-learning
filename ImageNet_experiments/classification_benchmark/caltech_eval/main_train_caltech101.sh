python main_caltech101.py \
  -a resnet50 \
  --lr 0.001 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --gpus 8,9,10,11 \
  --num-classes 102 \
  --save-dir caltech_checkpoints \
  --pretrained ./checkpoint.pth.tar \
  /opt/caoyh/datasets/caltech101/v1