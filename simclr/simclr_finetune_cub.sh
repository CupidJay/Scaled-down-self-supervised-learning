python main_simclr_finetune.py \
  -a resnet50 \
  --lr 0.5 \
  --batch-size 512 --epochs 200 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 8,9,10,11 \
  --mlp --simclr-t 0.1 --aug-plus --cos \
  --pretrained checkpoints/simclr_cub_800ep_bs_512_lr_0.5_56x56_pretrain.pth.tar \
  /opt/caoyh/datasets/cub200