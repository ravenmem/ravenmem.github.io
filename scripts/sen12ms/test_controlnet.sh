python test_controlnet.py \
    --dataset sen12ms \
    --sen12ms_root "./sen12ms" \
    --checkpoint_dir "./checkpoints/sen12ms/controlnet/checkpoint-100000" \
    --base_model_path "./stable-diffusion-2-1-base/stable-diffusion-2-1-base" \
    --dino_weights "./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
    --dino_checkpoint "./checkpoints/sen12ms/stage1_sar/checkpoint_stage1_epoch100.pth" \
    --output_dir "./validation_results/sen12ms" \
    --num_samples 3000