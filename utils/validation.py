"""
Validation utilities for SAR-to-Optical training.
"""

import os
import gc
import shutil
import logging

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from torch_fidelity import calculate_metrics

from utils.visualization import tensor_to_pil
from utils.prompts import logits_to_prompt, SD_V2_CLASS_PROMPTS
from utils.pipeline import load_seesr_pipeline

logger = logging.getLogger(__name__)


def run_validation(
    args,
    accelerator,
    global_step,
    unet,
    controlnet,
    classifier_model,
    vae,
    noise_scheduler,
    tokenizer,
    encoder_hidden_states,
    validation_dataloader,
    normalize_transform,
    weight_dtype,
):
    """
    Run validation loop: generate images and compute metrics.

    Args:
        args: Training arguments
        accelerator: Accelerator instance
        global_step: Current training step
        unet: UNet model (unwrapped)
        controlnet: ControlNet model (unwrapped)
        classifier_model: DINOv3 classifier
        vae: VAE model
        noise_scheduler: Noise scheduler
        tokenizer: Text tokenizer
        encoder_hidden_states: Cached text encoder states
        validation_dataloader: Validation data loader
        normalize_transform: Normalization transform
        weight_dtype: Model weight dtype

    Returns:
        Dict with 'avg_psnr' and 'fid_score'
    """
    import wandb

    valid_weight_dtype = torch.float32
    torch.cuda.empty_cache()
    gc.collect()

    image_logs_list = []

    with torch.no_grad():
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(args.validation_seed)

        pipeline = load_seesr_pipeline(
            args,
            accelerator,
            True,
            global_step,
            unet,
            controlnet
        )

        psnr_scores = []
        fid_real_path = os.path.join(args.output_dir, "fid_real_images")
        fid_fake_path = os.path.join(args.output_dir, "fid_fake_images")

        # Setup FID directories
        if accelerator.is_main_process:
            if os.path.exists(fid_real_path):
                shutil.rmtree(fid_real_path)
            if os.path.exists(fid_fake_path):
                shutil.rmtree(fid_fake_path)
            os.makedirs(fid_real_path, exist_ok=True)
            os.makedirs(fid_fake_path, exist_ok=True)

        for step, (batch, lbl, md_norm, seasons) in enumerate(validation_dataloader):
            if step >= args.validation_max_batches:
                break

            opt_image_256 = batch["opt"].to(accelerator.device, dtype=valid_weight_dtype)
            sar_image_256 = batch["sar"].to(accelerator.device, dtype=valid_weight_dtype)

            sar_classifier_cond = normalize_transform(sar_image_256).to(
                accelerator.device, dtype=weight_dtype
            )
            md_norm = md_norm.to(accelerator.device)

            # Extract features
            with torch.no_grad():
                logits, visual_features_dict = classifier_model(sar_classifier_cond)
                sorted_layer_indices = sorted(visual_features_dict.keys())

                feature_stack = []
                for idx in sorted_layer_indices:
                    feat = visual_features_dict[idx]["patch"]
                    feat = feat.flatten(2).transpose(1, 2)
                    feature_stack.append(feat)

            image_encoder_hidden_states = torch.stack(feature_stack, dim=0)
            image_encoder_hidden_states = image_encoder_hidden_states.to(
                accelerator.device, dtype=valid_weight_dtype
            )

            # Generate prompts
            prompts = logits_to_prompt(
                args=args,
                is_train=False,
                logits=logits,
                class_names=SD_V2_CLASS_PROMPTS,
                seasons=seasons,
                threshold=args.prompt_threshold,
                max_classes=args.prompt_max_classes
            )

            negative_prompt = getattr(args, 'negative_prompt',
                "low quality, worst quality, blurry, noisy, jpeg artifacts, "
                "speckle, speckle noise, grainy, monochrome, grayscale, dark"
            )
            negative_prompt_list = [negative_prompt] * len(prompts)

            # Generate images
            output = pipeline(
                prompts,
                sar_image_256,
                num_inference_steps=args.validation_inference_steps,
                generator=generator,
                guidance_scale=args.validation_guidance_scale,
                negative_prompt=negative_prompt_list,
                conditioning_scale=1.0,
                start_point="noise",
                ram_encoder_hidden_states=image_encoder_hidden_states,
                latent_tiled_size=9999,
                output_type="pil",
                metadata=md_norm
            )

            # Get confidence maps
            with torch.no_grad():
                opt_image_vae = opt_image_256 * 2.0 - 1.0
                latents_val = vae.encode(opt_image_vae.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents_val = latents_val * vae.config.scaling_factor

                bsz = latents_val.shape[0]
                t_conf = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents_val.device,
                ).long()

                noise_val = torch.randn_like(latents_val)
                noisy_latents_val = noise_scheduler.add_noise(latents_val, noise_val, t_conf)

                down_block_res_val, mid_block_res_val = controlnet(
                    noisy_latents_val,
                    t_conf,
                    metadata=md_norm,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=sar_image_256,
                    return_dict=False,
                    image_encoder_hidden_states=image_encoder_hidden_states,
                    is_multiscale_latent=True,
                )

                out_val = unet(
                    noisy_latents_val,
                    t_conf,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_block_res_val],
                    mid_block_additional_residual=mid_block_res_val.to(dtype=weight_dtype),
                    image_encoder_hidden_states=image_encoder_hidden_states,
                    is_multiscale_latent=True,
                    return_dict=True
                )
                noise_pred_val, conf_val = out_val[0], out_val[1]

            generated_images_256 = output.images

            # Calculate metrics per image
            for i, gen_img_256 in enumerate(generated_images_256):
                idx_str = f"{step * args.train_batch_size + i:05d}"

                gt_pil_256 = tensor_to_pil(opt_image_256[i])
                gt_np_256 = np.array(gt_pil_256)

                cond_pil_256 = tensor_to_pil(sar_image_256[i])
                gen_np_256 = np.array(gen_img_256)

                psnr = calculate_psnr(gt_np_256, gen_np_256, data_range=255)
                psnr_scores.append(psnr)

                # Save for FID calculation
                gen_img_256.save(os.path.join(fid_fake_path, f"{idx_str}.png"))
                gt_pil_256.save(os.path.join(fid_real_path, f"{idx_str}.png"))

                # Create confidence map visualization
                conf_map = conf_val[i, 0].detach().cpu().numpy()
                conf_min = conf_map.min()
                conf_max = conf_map.max()
                if conf_max > conf_min:
                    conf_vis = (conf_map - conf_min) / (conf_max - conf_min)
                else:
                    conf_vis = np.zeros_like(conf_map)
                conf_vis_img = (conf_vis * 255).astype(np.uint8)
                conf_pil = Image.fromarray(conf_vis_img)

                # Log to wandb
                image_logs = {
                    f"validation_samples-{idx_str}": [
                        wandb.Image(gen_np_256, caption="Generated (256px)"),
                        wandb.Image(np.array(cond_pil_256), caption="Input SAR (256px)"),
                        wandb.Image(gt_np_256, caption="GT Optical (256px)"),
                        wandb.Image(conf_pil, caption="Confidence Map"),
                    ]
                }
                accelerator.log(image_logs, step=global_step)

        # Calculate final metrics
        avg_psnr = np.mean(psnr_scores)
        fid_score = 0.0

        if accelerator.is_main_process:
            metrics_dict = calculate_metrics(
                input1=fid_real_path,
                input2=fid_fake_path,
                cuda=True, isc=False, fid=True, kid=False, prc=False
            )
            fid_score = metrics_dict['frechet_inception_distance']

            # Cleanup
            shutil.rmtree(fid_real_path)
            shutil.rmtree(fid_fake_path)

        logger.info(f"Validation Average PSNR: {avg_psnr:.4f} dB")
        logger.info(f"Validation FID: {fid_score:.4f}")

        torch.cuda.empty_cache()

    del pipeline
    torch.cuda.empty_cache()

    return {"avg_psnr": avg_psnr, "fid_score": fid_score}
