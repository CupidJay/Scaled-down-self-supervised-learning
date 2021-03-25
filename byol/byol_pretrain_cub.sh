python main_byol_finetune.py \
  -a resnet50 \
  --lr 0.125 \
  --batch-size 128 --epochs 800 \
  --input-size 56 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 12,13,14,15 \
  --byol-hidden-dim 2048 \
  --aug-plus --cos \
  /opt/caoyh/datasets/cub200