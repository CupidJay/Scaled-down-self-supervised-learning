python main.py \
  -a resnet50 \
  --lr 10.0 \
  --linear-eval \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained resnet50 \
  --gpus 4,5,6,7 \
  --num-classes 100 \
  /opt/caoyh/datasets/aircraft