GPUS="0,1"
export CUDA_VISIBLE_DEVICES=$GPUS

accelerate launch train_controlnet.py \
--dataset benv2 \
--pretrained_model_name_or_path="./stable-diffusion-2-1-base/stable-diffusion-2-1-base" \
--output_dir="/mnt/e/sar2opt/output_meta_cond_no_nh_np" \
--mixed_precision="bf16" \
--learning_rate=5e-5 \
--train_batch_size=8 \
--gradient_accumulation_steps=4 \
--checkpointing_steps=1000 \
--max_train_steps=100000 \
--report_to wandb \
--lr_scheduler "cosine" \
--lr_warmup_steps=100 \
--warmup_steps=10 \
--gradient_checkpointing \
--use_8bit_adam \
--set_grads_to_none \
--enable_xformers_memory_efficient_attention \
--trainable_modules "image_attentions" "conv_out_conf"