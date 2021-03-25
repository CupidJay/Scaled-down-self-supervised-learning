python main_simclr_pretraining.py \
  -a resnet50 \
  --lr 0.5 \
  --batch-size 512 --epochs 200 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --mlp --simclr-t 0.1 --aug-plus --cos \
  /opt/caoyh/datasets/cub200