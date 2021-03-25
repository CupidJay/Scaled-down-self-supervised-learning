python main_imagenet.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 256 --epochs 800 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cos --eval-freq 100 \
  /mnt/ramdisk/small_imagenet_total_50000