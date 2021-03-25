python main.py \
  -a resnet50 \
  --lr 10.0 \
  --linear-eval \
  --batch-size 64 --epochs 30 \
  --step-lr --schedule 10 20 \
  --gpus 12,13,14,15 \
  --num-classes 200 \
  --pretrained /opt/caoyh/code/SSL/random_potential/checkpoints/transform_2_stage/co_teaching_v2/resnet18_resnet50/gpus_4_lr_0.03_bs_128_epochs_200_path_cub200/checkpoint_0199.pth.tar \
  /opt/caoyh/datasets/cub200