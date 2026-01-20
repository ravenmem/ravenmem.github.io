"""
Pipeline utilities for loading and saving SeeSR models.
"""

import os
import logging
import torch

from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from utils.visualization import image_grid

logger = logging.getLogger(__name__)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention,
                        global_step, unet_model, controlnet_model):
    """
    Create inference pipeline from trained UNet and ControlNet.

    Args:
        args: Training arguments
        accelerator: Accelerator instance
        enable_xformers_memory_efficient_attention: Whether to enable xformers
        global_step: Current training step (for logging)
        unet_model: Trained UNet
        controlnet_model: Trained ControlNet

    Returns:
        StableDiffusionControlNetPipeline instance
    """
    from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_name_or_path}/feature_extractor")

    unet = unet_model
    controlnet = controlnet_model

    unet.eval()
    controlnet.eval()

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        unet=unet, controlnet=controlnet, scheduler=scheduler,
        safety_checker=None, requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device, torch.float32)
    vae.to(accelerator.device, torch.float32)
    pipeline.set_progress_bar_config(disable=True)

    logger.debug(f"Pipeline loaded - VAE: {pipeline.vae.dtype}, UNet: {pipeline.unet.dtype}")

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    unet.train()
    controlnet.train()

    return pipeline


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    """
    Generate HuggingFace model card README.

    Args:
        repo_id: Repository ID
        image_logs: Optional list of validation image logs
        base_model: Base model name
        repo_folder: Output folder path
    """
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)
