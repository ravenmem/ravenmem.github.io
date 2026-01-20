CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \
    --dataset sen12ms \
    --stage 1 \
    --data_type sar \
    --teacher_checkpoint ./checkpoints/stage0_opt/checkpoint_stage0_epoch100.pth \
    --output_dir ./checkpoints/stage1_sar \
    --sen12ms_root_dir ./sen12ms \
    --dinov3_repo /root/hyun/dinov3 --dinov3_pretrained_weights /root/hyun/현서/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth 

    