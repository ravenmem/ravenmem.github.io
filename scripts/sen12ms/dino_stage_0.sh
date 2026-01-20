CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \
    --dataset sen12ms \
    --stage 0 \
    --data_type opt \
    --sen12ms_root_dir ./sen12ms \
    --output_dir ./checkpoints/stage0_sar \
    --dinov3_repo /root/hyun/dinov3 --dinov3_pretrained_weights /root/hyun/현서/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth