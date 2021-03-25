python main_moco_finetune.py \
  -a resnet50 \
  --lr 0.3 \
  --batch-size 256 --epochs 800 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --moco-k 32768 \
  /mnt/ramdisk/small_imagenet_total_50000
