python main.py \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 64 --epochs 30 \
  --step-lr --schedule 10 20 \
  --pretrained /opt/caoyh/code/SSL/random_potential/checkpoints/finetune/224/custom_resnet50_custom_vgg16/gpus_4_lr_0.001_bs_128_epochs_200_path_cub200/custom_resnet50_checkpoint_0049.pth.tar \
  --gpus 4,5,6,7 \
  --num-classes 200 \
  /opt/caoyh/datasets/cub200