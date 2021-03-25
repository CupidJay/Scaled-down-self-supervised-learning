python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 --epochs 100\
  --pretrained detection/moco_v2_200ep_pretrain_112x112.pth.tar \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  /mnt/ramdisk/ImageNet