python main.py \
  -a resnet50 \
  --lr 10.0 \
  --linear-eval \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --pretrained resnet50 \
  --gpus 0,1,2,3 \
  --num-classes 200 \
  /opt/caoyh/datasets/cub200