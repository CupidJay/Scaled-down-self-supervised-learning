python main_freeze_voc.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 64 --epochs 60 \
  --step-lr --schedule 20 40 \
  --gpus 0,1,2,3 \
  --freeze \
  --pretrained /mnt/data3/caoyh/SSL/moco_v2_custom_res50_200ep_pretrain_112x112.pth.tar \
  --num-classes 20 \
  /opt/Dataset