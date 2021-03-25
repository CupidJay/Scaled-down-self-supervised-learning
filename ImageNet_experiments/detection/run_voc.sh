CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_net.py --config-file configs/pascal_voc_R_50_FPN_24k_moco.yaml --num-gpus 8 \
MODEL.WEIGHTS ./moco_v2_smallIN_total_50000_800ep_lr_0.3_k_32768_pretrain_112x112_output.pkl \
OUTPUT_DIR outputs/R50_FPN2/moco_v2_smallIN_total_50000_800ep_lr_0.3_k_32768_pretrain_112x112_output_voc \
