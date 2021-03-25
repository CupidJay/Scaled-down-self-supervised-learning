cp /opt/caoyh/code/SSL/COCO_codebase/moco/checkpoints/finetune/112/resnet50/gpus_8_lr_0.3_to_0.0_bs_256_epochs_800_path_small_imagenet_total_50000/checkpoint_0799.pth.tar \
./moco_v2_smallIN_total_50000_800ep_lr_0.3_k_32768_pretrain_112x112.pth.tar

python convert-pretrain-to-detectron2.py \
./moco_v2_smallIN_total_50000_800ep_lr_0.3_k_32768_pretrain_112x112.pth.tar ./moco_v2_smallIN_total_50000_800ep_lr_0.3_k_32768_pretrain_112x112_output.pkl
