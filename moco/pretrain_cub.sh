python main_moco_pretraining.py \
  -a custom_resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  /opt/caoyh/datasets/cub200
