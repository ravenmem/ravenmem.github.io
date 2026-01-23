"""
ControlNet LightningModule for SAR-to-Optical Image Synthesis.

This module implements diffusion-based SAR-to-Optical translation using:
- Pretrained Stable Diffusion 2.1 as the base model
- ControlNet for conditioning on SAR images
- DINO encoder (from Stage 1) for cross-attention features
"""

import math
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchvision.transforms import v2

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from utils.models import ViTKDDistillationModel, import_model_class_from_model_name_or_path, zero_init_image_attentions
from utils.prompts import SD_V2_CLASS_PROMPTS, LCCS_LU_CLASS_PROMPTS_V2, logits_to_prompt


class ControlNetModule(pl.LightningModule):
    """
    PyTorch Lightning Module for ControlNet-based SAR-to-Optical synthesis.

    This module handles:
    - Loading pretrained diffusion components (VAE, text encoder, tokenizer)
    - Loading frozen DINO encoder from Stage 1 checkpoint
    - ControlNet + UNet training with configurable trainable modules
    - Confidence-weighted diffusion loss

    Args:
        cfg: OmegaConf configuration object
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Dataset-specific configuration
        if cfg.data.dataset == "sen12ms":
            self.num_classes = 11
            self.class_prompts = LCCS_LU_CLASS_PROMPTS_V2
        else:  # benv2
            self.num_classes = cfg.model.classifier.num_classes
            self.class_prompts = SD_V2_CLASS_PROMPTS

        # Build models
        self._load_pretrained_models()
        self._load_dino_classifier()
        self._setup_controlnet()

        # Normalization for DINO input
        self.normalize = v2.Normalize(
            mean=cfg.data.preprocessing.norm_mean,
            std=cfg.data.preprocessing.norm_std
        )

        # Manual optimization for multiple parameter groups
        self.automatic_optimization = False

    def _load_pretrained_models(self):
        """Load pretrained Stable Diffusion components."""
        cfg = self.cfg.model
        pretrained_path = cfg.pretrained_model_path

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_path,
            subfolder="tokenizer"
        )

        # Text encoder
        text_encoder_cls = import_model_class_from_model_name_or_path(
            pretrained_path, cfg.revision
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_path,
            subfolder="text_encoder",
            revision=cfg.revision
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_path,
            subfolder="vae",
            revision=cfg.revision
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler"
        )

    def _load_dino_classifier(self):
        """Load frozen DINO encoder from Stage 1 Lightning checkpoint."""
        cfg = self.cfg.model.dino

        # Build backbone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backbone = torch.hub.load(
                cfg.repo_path,
                'dinov3_vitl16',
                source='local',
                weights=cfg.weights
            )

        self.dino_hidden_dim = backbone.embed_dim

        # Build classifier model
        self.classifier = ViTKDDistillationModel(
            backbone=backbone,
            num_classes=self.num_classes,
            layers=cfg.layers_to_extract
        )

        # Apply LoRA (matching Stage 1 config)
        lora_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
        )
        self.classifier = get_peft_model(self.classifier, lora_config)

        # Load from checkpoint
        checkpoint = torch.load(cfg.checkpoint, map_location='cpu')

        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = {
                k.replace('student_model.', ''): v
                for k, v in checkpoint['state_dict'].items()
                if k.startswith('student_model.')
            }
        elif 'model_state_dict' in checkpoint:
            # Legacy format
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.classifier.load_state_dict(state_dict, strict=False)

        # Freeze classifier
        self.classifier.requires_grad_(False)
        self.classifier.eval()

    def _setup_controlnet(self):
        """Setup ControlNet and UNet models."""
        cfg = self.cfg.model
        pretrained_path = cfg.pretrained_model_path

        # Load UNet
        if cfg.unet_model_path:
            self.unet = UNet2DConditionModel.from_pretrained_orig(
                pretrained_model_path=pretrained_path,
                seesr_model_path=cfg.unet_model_path,
                subfolder="unet",
                revision=cfg.revision,
                use_image_cross_attention=True,
                image_cross_attention_dim=self.dino_hidden_dim,
            )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_path,
                subfolder="unet",
                revision=cfg.revision,
                use_image_cross_attention=True,
                low_cpu_mem_usage=False,
                image_cross_attention_dim=self.dino_hidden_dim,
            )

        # Load ControlNet
        if cfg.controlnet_model_path:
            self.controlnet = ControlNetModel.from_pretrained(
                cfg.controlnet_model_path,
                subfolder="controlnet"
            )
        else:
            self.controlnet = ControlNetModel.from_unet(
                self.unet,
                use_image_cross_attention=True,
                image_cross_attention_dim=self.dino_hidden_dim,
            )

        # Zero-initialize image attention projections
        zero_init_image_attentions(self.unet, "UNet")
        zero_init_image_attentions(self.controlnet, "ControlNet")

        # Freeze base models
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Enable training for specified modules
        self._setup_trainable_modules()

        # Enable gradient checkpointing if configured
        if self.cfg.training.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()

    def _setup_trainable_modules(self):
        """Configure which modules should be trainable."""
        trainable_modules = self.cfg.training.trainable_modules

        # ControlNet
        if "controlnet" in trainable_modules:
            self.controlnet.train()
            self.controlnet.requires_grad_(True)
        else:
            self.controlnet.eval()
            self.controlnet.requires_grad_(False)

        # UNet
        if "unet" in trainable_modules:
            self.unet.train()
            self.unet.requires_grad_(True)
        else:
            self.unet.eval()

        # Text encoder
        if "text_encoder" in trainable_modules:
            self.text_encoder.train()
            self.text_encoder.requires_grad_(True)

        # Image attentions only (subset of UNet)
        if "image_attentions" in trainable_modules:
            for name, module in self.unet.named_modules():
                if name.endswith(("image_attentions",)):
                    for params in module.parameters():
                        params.requires_grad = True

    def _extract_dino_features(self, sar_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINO features from SAR images for cross-attention.

        Args:
            sar_images: SAR input tensor (B, 3, H, W)

        Returns:
            Tuple of (image_encoder_hidden_states, logits)
        """
        sar_normalized = self.normalize(sar_images)

        with torch.no_grad():
            logits, visual_features_dict = self.classifier(sar_normalized)

        # Stack features from all layers
        sorted_layer_indices = sorted(visual_features_dict.keys())
        feature_stack = []

        for idx in sorted_layer_indices:
            patch_tokens_map = visual_features_dict[idx]["patch"]
            cls_token = visual_features_dict[idx]["cls"]

            # Flatten patch tokens: [B, C, H, W] -> [B, N, C]
            patch_tokens_seq = patch_tokens_map.flatten(2).transpose(1, 2)

            # Add CLS token: [B, 1, C]
            cls_token_seq = cls_token.unsqueeze(1)

            # Concatenate: [B, 1+N, C]
            full_sequence = torch.cat((cls_token_seq, patch_tokens_seq), dim=1)
            feature_stack.append(full_sequence)

        # Stack all layers: [num_layers, B, 1+N, C]
        image_encoder_hidden_states = torch.stack(feature_stack, dim=0)

        return image_encoder_hidden_states, logits

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        inputs = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]

        return encoder_hidden_states

    def forward(
        self,
        sar_images: torch.Tensor,
        opt_images: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        seasons: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        weight_dtype = self.vae.dtype

        # Get DINO features
        image_encoder_hidden_states, logits = self._extract_dino_features(sar_images)
        image_encoder_hidden_states = image_encoder_hidden_states.to(dtype=weight_dtype)

        # Generate prompts from classifier output
        prompts = logits_to_prompt(
            args=self.cfg.prompts,
            is_train=self.training,
            logits=logits,
            class_names=self.class_prompts,
            seasons=seasons,
            threshold=self.cfg.prompts.threshold,
            max_classes=self.cfg.prompts.max_classes
        )

        # Encode prompts
        encoder_hidden_states = self._encode_prompts(prompts)

        # Encode optical images to latent space
        opt_vae_input = opt_images * 2.0 - 1.0
        with torch.no_grad():
            latents = self.vae.encode(opt_vae_input).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Add noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Apply metadata dropout if configured
        if metadata is not None and self.cfg.losses.metadata_dropout > 0:
            keep_mask = torch.rand_like(metadata) > self.cfg.losses.metadata_dropout
            metadata = metadata * keep_mask

        # ControlNet forward
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            metadata=metadata,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=sar_images,
            return_dict=False,
            image_encoder_hidden_states=image_encoder_hidden_states,
            is_multiscale_latent=True
        )

        # UNet forward
        out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            image_encoder_hidden_states=image_encoder_hidden_states,
            is_multiscale_latent=True,
            return_dict=True,
        )

        model_pred = out[0]
        confidence = out[1]

        # Get target based on prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        return {
            "model_pred": model_pred,
            "target": target,
            "confidence": confidence,
            "timesteps": timesteps,
        }

    def _compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute confidence-weighted diffusion loss."""
        loss_beta = self.cfg.losses.loss_beta
        tau = math.sqrt(math.log(2 * math.pi))

        # Confidence-weighted reconstruction loss
        weighted_residual = (target - model_pred) * (confidence ** loss_beta)
        loss_recon = (weighted_residual ** 2).mean()

        # Regularization loss
        loss_reg = -torch.log(confidence ** loss_beta + 1e-8).mean() + (tau ** 2) / 2.0

        total_loss = loss_recon + loss_reg

        return {
            "loss": total_loss,
            "loss_recon": loss_recon,
            "loss_reg": loss_reg,
        }

    def training_step(self, batch: Tuple, batch_idx: int):
        """Training step with manual optimization."""
        # Unpack batch
        if len(batch) == 4:
            img_dict, labels, metadata, seasons = batch
        elif len(batch) == 3:
            img_dict, labels, _ = batch
            metadata, seasons = None, None
        else:
            img_dict, labels = batch
            metadata, seasons = None, None

        opt_images = img_dict["opt"]
        sar_images = img_dict["sar"]

        # Forward pass
        outputs = self.forward(sar_images, opt_images, labels, metadata, seasons)

        # Compute loss
        losses = self._compute_loss(
            outputs["model_pred"],
            outputs["target"],
            outputs["confidence"]
        )

        # Manual optimization
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad(set_to_none=self.cfg.training.set_grads_to_none)
        self.manual_backward(losses["loss"])

        # Gradient clipping
        if self.cfg.training.max_grad_norm > 0:
            params_to_clip = list(self.controlnet.parameters()) + list(self.unet.parameters())
            self.clip_gradients(
                opt,
                gradient_clip_val=self.cfg.training.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt.step()
        sch.step()

        # Log metrics
        self.log_dict({
            "train/loss": losses["loss"],
            "train/loss_recon": losses["loss_recon"],
            "train/loss_reg": losses["loss_reg"],
            "train/lr": sch.get_last_lr()[0],
        }, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)

        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Periodic memory cleanup."""
        if batch_idx % 100 == 0 and self.global_rank == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step - compute loss on validation data."""
        # Unpack batch
        if len(batch) == 4:
            img_dict, labels, metadata, seasons = batch
        elif len(batch) == 3:
            img_dict, labels, _ = batch
            metadata, seasons = None, None
        else:
            img_dict, labels = batch
            metadata, seasons = None, None

        opt_images = img_dict["opt"]
        sar_images = img_dict["sar"]

        # Forward pass
        outputs = self.forward(sar_images, opt_images, labels, metadata, seasons)

        # Compute loss
        losses = self._compute_loss(
            outputs["model_pred"],
            outputs["target"],
            outputs["confidence"]
        )

        # Log metrics
        self.log_dict({
            "val/loss": losses["loss"],
            "val/loss_recon": losses["loss_recon"],
            "val/loss_reg": losses["loss_reg"],
        }, sync_dist=True)

        return losses["loss"]

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        cfg = self.cfg.training

        # Collect trainable parameters
        params_to_optimize = []

        if "controlnet" in cfg.trainable_modules:
            params_to_optimize += list(self.controlnet.parameters())
        if "unet" in cfg.trainable_modules:
            params_to_optimize += list(self.unet.parameters())
        if "text_encoder" in cfg.trainable_modules:
            params_to_optimize += list(self.text_encoder.parameters())
        if "image_attentions" in cfg.trainable_modules:
            for name, module in self.unet.named_modules():
                if name.endswith(("image_attentions",)):
                    params_to_optimize += list(module.parameters())

        # Filter for parameters that require grad
        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]

        # Use 8-bit Adam if configured
        if cfg.optimizer.use_8bit:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("8-bit Adam requires bitsandbytes: pip install bitsandbytes")
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.epsilon,
        )

        # Calculate total training steps
        if cfg.max_train_steps:
            num_training_steps = cfg.max_train_steps
        else:
            # Estimate from trainer
            num_training_steps = self.trainer.estimated_stepping_batches

        # Learning rate scheduler
        scheduler = get_scheduler(
            cfg.scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=cfg.scheduler.warmup_steps * self.trainer.num_devices,
            num_training_steps=num_training_steps * self.trainer.num_devices,
            num_cycles=cfg.scheduler.num_cycles,
            power=cfg.scheduler.power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def on_train_start(self):
        """Setup dtype for inference components."""
        weight_dtype = torch.float32
        if self.trainer.precision == "16-mixed" or self.trainer.precision == "fp16":
            weight_dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed" or self.trainer.precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move frozen models to appropriate dtype
        self.classifier.to(dtype=weight_dtype)
        self.vae.to(dtype=weight_dtype)
        self.text_encoder.to(dtype=weight_dtype)
