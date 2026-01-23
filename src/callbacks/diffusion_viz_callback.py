"""
Diffusion visualization callback for ControlNet SAR-to-Optical synthesis.

This callback generates and logs visualizations during training to
monitor the quality of SAR-to-Optical translation.
"""

import os
import gc
import shutil
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import io

from utils.visualization import tensor_to_pil
from utils.prompts import logits_to_prompt


class DiffusionVisualizationCallback(Callback):
    """
    Callback for generating diffusion model visualizations.

    Periodically generates optical images from SAR inputs and logs:
    - Side-by-side comparisons (SAR input, generated optical, ground truth optical)
    - Confidence maps
    - Quantitative metrics (PSNR, SSIM)

    Args:
        vis_interval: Number of steps between visualizations
        num_samples: Number of samples to visualize
        inference_steps: Number of denoising steps for generation
        guidance_scale: Classifier-free guidance scale
        save_dir: Directory to save visualization images
    """

    def __init__(
        self,
        vis_interval: int = 500,
        num_samples: int = 4,
        inference_steps: int = 50,
        guidance_scale: float = 5.5,
        save_dir: str = "./vis_results/diffusion"
    ):
        super().__init__()
        self.vis_interval = vis_interval
        self.num_samples = num_samples
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.save_dir = save_dir

        self.fixed_val_batch = None
        self.pipeline = None
        self.generator = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Fetch fixed validation batch for consistent visualization."""
        if trainer.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)

            # Get validation batch
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))

            # Unpack batch based on format
            if len(batch) == 4:
                img_dict, labels, metadata, seasons = batch
            elif len(batch) == 3:
                img_dict, labels, _ = batch
                metadata, seasons = None, None
            else:
                img_dict, labels = batch
                metadata, seasons = None, None

            # Store fixed batch for consistent visualization
            self.fixed_val_batch = {
                "sar": img_dict["sar"][:self.num_samples].to(pl_module.device),
                "opt": img_dict["opt"][:self.num_samples].to(pl_module.device),
                "labels": labels[:self.num_samples].to(pl_module.device) if labels is not None else None,
                "metadata": metadata[:self.num_samples].to(pl_module.device) if metadata is not None else None,
                "seasons": seasons[:self.num_samples] if seasons is not None else None,
            }

            # Create generator for reproducible sampling
            self.generator = torch.Generator(device=pl_module.device)
            self.generator.manual_seed(42)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log visualizations at the end of validation."""
        if trainer.global_rank != 0:
            return

        if self.fixed_val_batch is None:
            return

        try:
            self._generate_visualizations(trainer, pl_module)
        except Exception as e:
            print(f"Diffusion visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _generate_visualizations(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Generate and log diffusion visualizations."""
        # Build inference pipeline
        pipeline = self._create_pipeline(pl_module)

        if pipeline is None:
            print("Could not create inference pipeline, skipping visualization")
            return

        # Set models to eval
        pipeline.controlnet.eval()
        pipeline.unet.eval()

        # Get DINO features and prompts
        sar_images = self.fixed_val_batch["sar"]
        opt_images = self.fixed_val_batch["opt"]
        labels = self.fixed_val_batch["labels"]
        metadata = self.fixed_val_batch["metadata"]
        seasons = self.fixed_val_batch["seasons"]

        with torch.no_grad():
            # Get weight dtype
            weight_dtype = pl_module.vae.dtype

            # Extract DINO features
            image_encoder_hidden_states, logits = pl_module._extract_dino_features(sar_images)
            image_encoder_hidden_states = image_encoder_hidden_states.to(dtype=weight_dtype)

            # Generate prompts
            prompts = logits_to_prompt(
                args=pl_module.cfg.prompts,
                is_train=False,
                logits=logits,
                class_names=pl_module.class_prompts,
                seasons=seasons,
                threshold=pl_module.cfg.prompts.threshold,
                max_classes=pl_module.cfg.prompts.max_classes
            )

            negative_prompt = [pl_module.cfg.prompts.negative_prompt] * len(prompts)

            # Generate images
            output = pipeline(
                prompts,
                sar_images,
                num_inference_steps=self.inference_steps,
                generator=self.generator,
                guidance_scale=self.guidance_scale,
                negative_prompt=negative_prompt,
                conditioning_scale=1.0,
                start_point="noise",
                ram_encoder_hidden_states=image_encoder_hidden_states,
                output_type="pil",
                metadata=metadata
            )

            generated_images = output.images

        # Create visualization
        vis_image = self._create_comparison_grid(
            sar_images, opt_images, generated_images, prompts, trainer
        )

        # Compute metrics
        metrics = self._compute_metrics(opt_images, generated_images)

        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            import wandb
            trainer.logger.experiment.log({
                "val/comparison_grid": wandb.Image(vis_image),
                "val/psnr": metrics["psnr"],
                "val/ssim": metrics["ssim"],
            }, step=trainer.global_step)

        # Save locally
        vis_image.save(os.path.join(self.save_dir, f"comparison_step_{trainer.global_step}.png"))

        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

        # Restore training mode
        pl_module.controlnet.train()
        pl_module.unet.train()

    def _create_pipeline(self, pl_module):
        """Create inference pipeline from trained models."""
        try:
            from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
            from diffusers import DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
            from diffusers import AutoencoderKL

            cfg = pl_module.cfg.model
            pretrained_path = cfg.pretrained_model_path

            scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder")
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
            vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
            feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_path}/feature_extractor")

            # Get unwrapped models from module
            unet = pl_module.unet
            controlnet = pl_module.controlnet

            # Create pipeline
            pipeline = StableDiffusionControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipeline = pipeline.to(pl_module.device, torch.float32)
            pipeline.vae.to(dtype=torch.float32)
            pipeline.set_progress_bar_config(disable=True)

            return pipeline

        except Exception as e:
            print(f"Failed to create pipeline: {e}")
            return None

    def _create_comparison_grid(
        self,
        sar_images: torch.Tensor,
        opt_images: torch.Tensor,
        generated_images: List[Image.Image],
        prompts: List[str],
        trainer: pl.Trainer
    ) -> Image.Image:
        """Create side-by-side comparison grid."""
        B = min(len(generated_images), self.num_samples)

        fig, axs = plt.subplots(B, 3, figsize=(15, 5 * B))
        fig.suptitle(
            f"SAR-to-Optical Generation (Step {trainer.global_step})",
            fontsize=16
        )

        if B == 1:
            axs = axs[np.newaxis, :]

        for idx in range(B):
            # SAR input
            sar_pil = tensor_to_pil(sar_images[idx])
            axs[idx, 0].imshow(sar_pil)
            if idx == 0:
                axs[idx, 0].set_title("SAR Input", fontsize=12)
            axs[idx, 0].axis('off')

            # Generated optical
            gen_img = generated_images[idx]
            axs[idx, 1].imshow(gen_img)
            if idx == 0:
                axs[idx, 1].set_title("Generated Optical", fontsize=12)
            axs[idx, 1].set_xlabel(prompts[idx][:50] + "..." if len(prompts[idx]) > 50 else prompts[idx], fontsize=8)
            axs[idx, 1].axis('off')

            # Ground truth optical
            gt_pil = tensor_to_pil(opt_images[idx])
            axs[idx, 2].imshow(gt_pil)
            if idx == 0:
                axs[idx, 2].set_title("Ground Truth Optical", fontsize=12)
            axs[idx, 2].axis('off')

        plt.tight_layout()

        # Convert to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        result_image = Image.open(buf).copy()
        buf.close()
        plt.close()

        return result_image

    def _compute_metrics(
        self,
        opt_images: torch.Tensor,
        generated_images: List[Image.Image]
    ) -> Dict[str, float]:
        """Compute PSNR and SSIM metrics."""
        psnr_scores = []
        ssim_scores = []

        for idx, gen_img in enumerate(generated_images):
            gt_pil = tensor_to_pil(opt_images[idx])
            gt_np = np.array(gt_pil)
            gen_np = np.array(gen_img)

            # Ensure same size
            if gen_np.shape != gt_np.shape:
                gen_img_resized = gen_img.resize((gt_np.shape[1], gt_np.shape[0]), Image.BICUBIC)
                gen_np = np.array(gen_img_resized)

            psnr = calculate_psnr(gt_np, gen_np, data_range=255)
            ssim = calculate_ssim(gt_np, gen_np, channel_axis=2, data_range=255)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

        return {
            "psnr": np.mean(psnr_scores),
            "ssim": np.mean(ssim_scores),
        }


class ConfidenceMapCallback(Callback):
    """
    Callback for logging confidence map visualizations.

    This callback visualizes the confidence maps predicted by the model,
    which indicate where the model is uncertain about its predictions.
    """

    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int
    ):
        """Log confidence map visualization."""
        if trainer.global_step % self.log_interval != 0:
            return

        if trainer.global_rank != 0:
            return

        # Get confidence from outputs if available
        # This requires modifying training_step to return confidence in outputs
        if isinstance(outputs, dict) and "confidence" in outputs:
            conf = outputs["confidence"]

            # Create visualization
            conf_np = conf[0, 0].detach().cpu().numpy()

            # Normalize for visualization
            conf_min, conf_max = conf_np.min(), conf_np.max()
            if conf_max > conf_min:
                conf_vis = (conf_np - conf_min) / (conf_max - conf_min)
            else:
                conf_vis = np.zeros_like(conf_np)

            conf_vis = (conf_vis * 255).astype(np.uint8)
            conf_pil = Image.fromarray(conf_vis)

            # Log to wandb
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                import wandb
                trainer.logger.experiment.log({
                    "train/confidence_map": wandb.Image(conf_pil),
                }, step=trainer.global_step)
