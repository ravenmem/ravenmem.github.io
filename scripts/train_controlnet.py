#!/usr/bin/env python
"""
Main training script for ControlNet SAR-to-Optical synthesis with PyTorch Lightning and Hydra.

Usage:
    # Train on BENv2
    python scripts/train_controlnet.py experiment=benv2

    # Train on SEN12MS
    python scripts/train_controlnet.py experiment=sen12ms

    # Override config values
    python scripts/train_controlnet.py experiment=benv2 training.optimizer.lr=1e-5

    # Multi-GPU training
    python scripts/train_controlnet.py experiment=benv2 trainer.devices=4

    # Debug mode
    python scripts/train_controlnet.py experiment=benv2 debug.fast_dev_run=true
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from src.modules.controlnet_module import ControlNetModule
from src.datamodules.multimodal_datamodule import MultimodalDataModule
from src.callbacks.diffusion_viz_callback import DiffusionVisualizationCallback


def validate_config(cfg: DictConfig):
    """Validate configuration before training."""
    # Check DINO checkpoint exists
    if not os.path.exists(cfg.model.dino.checkpoint):
        raise FileNotFoundError(
            f"DINO checkpoint not found: {cfg.model.dino.checkpoint}\n"
            "Please train Stage 1 DINO encoder first or provide correct path."
        )

    # Check pretrained model exists
    if not os.path.exists(cfg.model.pretrained_model_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {cfg.model.pretrained_model_path}\n"
            "Please download Stable Diffusion 2.1 base model."
        )

    # Validate trainable modules
    valid_modules = {"controlnet", "unet", "text_encoder", "image_attentions"}
    for module in cfg.training.trainable_modules:
        if module not in valid_modules:
            raise ValueError(
                f"Invalid trainable module: {module}. "
                f"Valid options: {valid_modules}"
            )


@hydra.main(version_base=None, config_path="../configs/controlnet", config_name="default")
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Validate config
    validate_config(cfg)

    # Set seed for reproducibility
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create output directory
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    # Initialize datamodule
    # Note: Reusing MultimodalDataModule with data_type override for paired training
    # Override data_type to get both SAR and optical in the same batch
    datamodule = MultimodalDataModule(cfg)

    # Initialize model
    model = ControlNetModule(cfg)

    # Setup logger with resume support
    logger = None
    if cfg.logging.wandb.enabled:
        wandb_resume_id = cfg.logging.wandb.get('resume_id', None)
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.experiment.name,
            save_dir=cfg.experiment.output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["controlnet", cfg.data.dataset],
            id=wandb_resume_id,
            resume="must" if wandb_resume_id else "allow",
        )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.experiment.output_dir,
            filename="checkpoint-{step:06d}",
            save_top_k=cfg.checkpoint.checkpoints_total_limit or 3,
            monitor="val/loss",
            mode="min",
            save_last=True,
            every_n_train_steps=cfg.checkpoint.save_every_n_steps,
        ),
        LearningRateMonitor(logging_interval="step"),
        DiffusionVisualizationCallback(
            vis_interval=cfg.validation.interval_steps,
            num_samples=cfg.validation.num_samples,
            inference_steps=cfg.validation.inference_steps,
            guidance_scale=cfg.validation.guidance_scale,
            save_dir=os.path.join(cfg.experiment.output_dir, "vis_results"),
        ),
    ]

    # Build trainer config
    trainer_config = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "strategy": cfg.trainer.strategy,
        "precision": cfg.trainer.precision,
        "max_epochs": cfg.trainer.max_epochs,
        "max_steps": cfg.training.max_train_steps or -1,
        "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
        "gradient_clip_val": cfg.trainer.gradient_clip_val,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "val_check_interval": cfg.trainer.val_check_interval,
        "enable_checkpointing": cfg.trainer.enable_checkpointing,
        "enable_progress_bar": cfg.trainer.enable_progress_bar,
        "deterministic": cfg.trainer.deterministic,
        "logger": logger,
        "callbacks": callbacks,
        "default_root_dir": cfg.experiment.output_dir,
    }

    # Handle debug options
    if cfg.debug.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    if cfg.debug.get("limit_train_batches", 1.0) < 1.0:
        trainer_config["limit_train_batches"] = cfg.debug.limit_train_batches
    if cfg.debug.get("limit_val_batches", 1.0) < 1.0:
        trainer_config["limit_val_batches"] = cfg.debug.limit_val_batches

    # Initialize trainer
    trainer = pl.Trainer(**trainer_config)

    # Print training info
    print(f"\nStarting ControlNet training:")
    print(f"  Dataset: {cfg.data.dataset}")
    print(f"  Output dir: {cfg.experiment.output_dir}")
    print(f"  Max epochs: {cfg.training.num_epochs}")
    print(f"  Max steps: {cfg.training.max_train_steps or 'unlimited'}")
    print(f"  Batch size: {cfg.data.dataloader.batch_size}")
    print(f"  Learning rate: {cfg.training.optimizer.lr}")
    print(f"  Trainable modules: {cfg.training.trainable_modules}")
    print(f"  DINO checkpoint: {cfg.model.dino.checkpoint}")
    print(f"  Pretrained model: {cfg.model.pretrained_model_path}")

    # Check for resume checkpoint
    ckpt_path = cfg.checkpoint.get('resume_path', None)
    if ckpt_path == "latest":
        # Find latest checkpoint
        ckpt_dir = cfg.experiment.output_dir
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            ckpt_path = os.path.join(ckpt_dir, checkpoints[-1])
        else:
            ckpt_path = None

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  Resuming from: {ckpt_path}")
    else:
        ckpt_path = None
    print()

    # Start training (with optional checkpoint resume)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("\n" + "=" * 60)
    print("Training completed!")
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 60)

    # Save final models
    if trainer.is_global_zero:
        final_dir = os.path.join(cfg.experiment.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        # Save ControlNet
        model.controlnet.save_pretrained(os.path.join(final_dir, "controlnet"))
        print(f"ControlNet saved to: {os.path.join(final_dir, 'controlnet')}")

        # Save UNet (if trainable)
        if "unet" in cfg.training.trainable_modules or "image_attentions" in cfg.training.trainable_modules:
            model.unet.save_pretrained(os.path.join(final_dir, "unet"))
            print(f"UNet saved to: {os.path.join(final_dir, 'unet')}")


if __name__ == "__main__":
    main()
