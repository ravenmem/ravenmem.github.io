CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \
    --dataset benv2 \
    --stage 0 \
    --data_type opt \
    --output_dir ./checkpoints/stage0_opt \
    --dinov3_repo /root/hyun/dinov3 \
    --dinov3_pretrained_weights /root/hyun/현서/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
    --dataset_images_lmdb /root/hyun/rico-hdl/Encoded-BigEarthNet --dataset_metadata_parquet /root/hyun/meta/metadata.parquet --dataset_metadata_snow_cloud_parquet /root/hyun/meta/metadata_for_patches_with_snow_cloud_or_shadow.parquet

