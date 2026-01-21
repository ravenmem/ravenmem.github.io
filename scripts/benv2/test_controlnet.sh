python test_controlnet.py \
    --dataset benv2 \
    --checkpoint_dir "./checkpoints/benv2/controlnet/checkpoint-100000" \
    --base_model_path "./stable-diffusion-2-1-base/stable-diffusion-2-1-base" \
    --dino_weights "./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
    --dino_checkpoint "./checkpoints/benv2/stage1_sar/checkpoint_stage1_epoch100.pth" \
    --images_lmdb /path/to/Encoded-BigEarthNet \
    --metadata_parquet /path/to/metadata.parquet \
    --metadata_snow_cloud_parquet /path/to/metadata_for_patches_with_snow_cloud_or_shadow.parquet \
    --output_dir "./validation_results/benv2" \
    --num_samples 1000



# python test_controlnet.py \
#     --dataset benv2 \
#     --checkpoint_dir "./checkpoints/yh_np" \
#     --base_model_path "./stable-diffusion-2-1-base/stable-diffusion-2-1-base" \
#     --dino_weights "./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
#     --dino_checkpoint "./checkpoints/12-04.pth" \
#     --output_dir "./validation_results/benv2" \
#     --num_samples 1000