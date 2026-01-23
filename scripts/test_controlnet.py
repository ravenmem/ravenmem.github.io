#!/usr/bin/env python
"""
Evaluation script for ControlNet SAR-to-Optical synthesis with Hydra.

Usage:
    # Evaluate on BENv2
    python scripts/test_controlnet.py experiment=benv2 \
        checkpoint.path=./checkpoints/controlnet/benv2/final

    # Evaluate on SEN12MS
    python scripts/test_controlnet.py experiment=sen12ms \
        checkpoint.path=./checkpoints/controlnet/sen12ms/final

    # Override evaluation settings
    python scripts/test_controlnet.py experiment=benv2 \
        checkpoint.path=./checkpoints/controlnet/benv2/final \
        validation.num_samples=1000 \
        validation.inference_steps=50
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import warnings

# Metrics
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torch_fidelity import calculate_metrics as calculate_fid_metrics
import lpips
from DISTS_pytorch import DISTS
from peft import LoraConfig, get_peft_model

# Models and utilities
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from torchvision.transforms import v2
from safetensors.torch import load_file

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from src.datamodules.multimodal_datamodule import MultimodalDataModule
from utils.visualization import tensor_to_pil
from utils.models import ViTKDDistillationModel
from utils.prompts import SD_V2_CLASS_PROMPTS, LCCS_LU_CLASS_PROMPTS_V2, logits_to_prompt
from utils.metrics import calculate_qnr_script_a, calculate_sam_metric, calculate_rmse, calculate_scc
from utils.dataloaders import ben_collate_fn, sen12ms_collate_fn


def load_weights_into_model(model, weight_path, model_name):
    """Load weights from safetensors file into model."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    print(f"Loading {model_name} from {os.path.basename(weight_path)}...")
    state_dict = load_file(weight_path)

    # Handle module. prefix from DDP training
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    m, u = model.load_state_dict(new_state_dict, strict=True)
    if len(m) > 0 or len(u) > 0:
        print(f"Warning: Missing keys: {len(m)}, Unexpected keys: {len(u)}")
    else:
        print(f"Successfully loaded {model_name}.")


@hydra.main(version_base=None, config_path="../configs/controlnet", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # Print config
    print("=" * 60)
    print("Evaluation Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Check checkpoint path
    checkpoint_path = cfg.checkpoint.get('path', None)
    if checkpoint_path is None:
        raise ValueError("checkpoint.path is required for evaluation")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Dataset-specific configuration
    if cfg.data.dataset == "sen12ms":
        NUM_CLASSES = 11
        CLASS_PROMPTS = LCCS_LU_CLASS_PROMPTS_V2
        print(f"Using SEN12MS dataset configuration: {NUM_CLASSES} classes")
    else:  # benv2
        NUM_CLASSES = cfg.model.classifier.num_classes
        CLASS_PROMPTS = SD_V2_CLASS_PROMPTS
        print(f"Using BENv2 dataset configuration: {NUM_CLASSES} classes")

    # Device and precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if cfg.trainer.precision == "bf16-mixed":
        weight_dtype = torch.bfloat16
    elif cfg.trainer.precision == "16-mixed":
        weight_dtype = torch.float16

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create output directory
    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    fid_real_path = os.path.join(output_dir, "fid_real")
    fid_fake_path = os.path.join(output_dir, "fid_fake")
    os.makedirs(fid_real_path, exist_ok=True)
    os.makedirs(fid_fake_path, exist_ok=True)

    # Initialize metric models
    print("Initializing metric models...")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()

    dists_loss_fn = DISTS().to(device)
    dists_loss_fn.eval()

    # Load DINO classifier
    print(f"Loading DINO classifier from {cfg.model.dino.checkpoint}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = torch.hub.load(
            cfg.model.dino.repo_path,
            'dinov3_vitl16',
            source='local',
            weights=cfg.model.dino.weights
        )

    classifier_model = ViTKDDistillationModel(
        backbone, num_classes=NUM_CLASSES, layers=cfg.model.dino.layers_to_extract
    )

    lora_config = LoraConfig(
        r=cfg.model.dino.lora.rank,
        lora_alpha=cfg.model.dino.lora.alpha,
        target_modules=cfg.model.dino.lora.target_modules,
        lora_dropout=cfg.model.dino.lora.dropout,
        bias=cfg.model.dino.lora.bias
    )
    classifier_model = get_peft_model(classifier_model, lora_config)

    # Load DINO checkpoint
    ckpt = torch.load(cfg.model.dino.checkpoint, map_location='cpu')
    if 'state_dict' in ckpt:
        # Lightning checkpoint
        state_dict = {
            k.replace('student_model.', ''): v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('student_model.')
        }
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    classifier_model.load_state_dict(state_dict, strict=False)
    classifier_model.to(device, dtype=weight_dtype)
    classifier_model.eval()

    # Load diffusion pipeline
    print(f"Loading pipeline from {checkpoint_path}...")
    pretrained_path = cfg.model.pretrained_model_path

    scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_path}/feature_extractor")

    dino_hidden_dim = backbone.embed_dim

    # Load UNet and ControlNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        use_image_cross_attention=True,
        image_encoder_hidden_states=dino_hidden_dim
    )
    controlnet = ControlNetModel.from_unet(
        unet,
        use_image_cross_attention=True,
        image_cross_attention_dim=dino_hidden_dim
    )

    # Load trained weights
    controlnet_path = os.path.join(checkpoint_path, "controlnet", "model.safetensors")
    unet_path = os.path.join(checkpoint_path, "unet", "model.safetensors")

    if os.path.exists(controlnet_path):
        load_weights_into_model(controlnet, controlnet_path, "ControlNet")
    else:
        # Try alternative path
        alt_controlnet = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(alt_controlnet):
            load_weights_into_model(controlnet, alt_controlnet, "ControlNet")

    if os.path.exists(unet_path):
        load_weights_into_model(unet, unet_path, "UNet")
    else:
        # Try alternative path
        alt_unet = os.path.join(checkpoint_path, "model_1.safetensors")
        if os.path.exists(alt_unet):
            load_weights_into_model(unet, alt_unet, "UNet")

    # Create pipeline
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        feature_extractor=feature_extractor, unet=unet, controlnet=controlnet,
        scheduler=scheduler, safety_checker=None, requires_safety_checker=False
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.to(dtype=torch.float32)

    if torch.cuda.is_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Setup dataset
    datamodule = MultimodalDataModule(cfg)
    datamodule.setup("test")
    val_dataset = datamodule.test_dataset

    # Create subset
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    num_samples = cfg.validation.num_samples or len(val_dataset)
    indices = torch.randperm(len(val_dataset), generator=generator)[:num_samples]
    val_subset = Subset(val_dataset, indices)

    collate_fn = sen12ms_collate_fn if cfg.data.dataset == "sen12ms" else ben_collate_fn
    val_dataloader = DataLoader(
        val_subset,
        batch_size=cfg.data.dataloader.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Normalization for DINO
    normalize_tf = v2.Normalize(
        mean=tuple(cfg.data.preprocessing.norm_mean),
        std=tuple(cfg.data.preprocessing.norm_std)
    )

    # Metrics containers
    metrics = {k: [] for k in ["psnr", "ssim", "lpips", "rmse", "sam_angle", "scc", "dists", "qnr", "d_lambda", "d_s"]}

    # Run evaluation
    print(f"Starting evaluation on {cfg.data.dataset.upper()} dataset...")
    print(f"Number of samples: {len(val_subset)}")
    generator_inf = torch.Generator(device=device).manual_seed(cfg.validation.seed)

    for step, batch_data in enumerate(tqdm(val_dataloader)):
        # Unpack batch
        if len(batch_data) == 4:
            batch, lbl, md_norm, seasons = batch_data
        elif len(batch_data) == 3:
            batch, lbl, _ = batch_data
            md_norm, seasons = None, None
        else:
            batch, lbl = batch_data
            md_norm, seasons = None, None

        sar_image = batch["sar"].to(device, dtype=weight_dtype)
        opt_image = batch["opt"].to(device, dtype=weight_dtype)

        # Extract DINO features
        sar_classifier_cond = normalize_tf(sar_image.float())

        with torch.no_grad():
            logits, visual_features_dict = classifier_model(sar_classifier_cond)

            feature_stack = []
            for idx in sorted(visual_features_dict.keys()):
                feat = visual_features_dict[idx]["patch"].flatten(2).transpose(1, 2)
                cls_token = visual_features_dict[idx]["cls"].unsqueeze(1)
                feature_stack.append(torch.cat((cls_token, feat), dim=1))
            image_encoder_hidden_states = torch.stack(feature_stack, dim=0).to(dtype=weight_dtype)

            # Generate prompts
            prompts = logits_to_prompt(
                args=cfg.prompts,
                is_train=False,
                logits=logits,
                class_names=CLASS_PROMPTS,
                seasons=seasons,
                threshold=cfg.prompts.threshold,
                max_classes=cfg.prompts.max_classes
            )

            negative_prompt = [cfg.prompts.negative_prompt] * len(prompts)

            # Generate images
            output = pipeline(
                prompts, sar_image.float(),
                num_inference_steps=cfg.validation.inference_steps,
                generator=generator_inf,
                guidance_scale=cfg.validation.guidance_scale,
                negative_prompt=negative_prompt,
                ram_encoder_hidden_states=image_encoder_hidden_states,
                output_type="pil",
                metadata=md_norm
            )
            generated_images = output.images

        # Calculate metrics
        for i, gen_img in enumerate(generated_images):
            idx_str = f"{step * cfg.data.dataloader.batch_size + i:05d}"

            gt_pil = tensor_to_pil(opt_image[i].float())
            gt_img_u8 = np.array(gt_pil)
            gen_img_np = np.array(gen_img)

            # Basic metrics
            metrics["psnr"].append(calculate_psnr(gt_img_u8, gen_img_np, data_range=255))
            metrics["ssim"].append(calculate_ssim(gt_img_u8, gen_img_np, channel_axis=2, data_range=255))
            metrics["rmse"].append(calculate_rmse(gt_img_u8, gen_img_np))
            metrics["scc"].append(calculate_scc(gt_img_u8, gen_img_np))
            metrics["sam_angle"].append(calculate_sam_metric(gt_img_u8, gen_img_np))

            # Deep learning metrics
            gt_t = torch.tensor(gt_img_u8).float().to(device).permute(2, 0, 1).unsqueeze(0)
            gen_t = torch.tensor(gen_img_np).float().to(device).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                metrics["lpips"].append(lpips_loss_fn((gen_t / 127.5) - 1, (gt_t / 127.5) - 1).item())
                metrics["dists"].append(dists_loss_fn(gen_t / 255.0, gt_t / 255.0).item())

            # QNR
            qnr_val, dl_val, ds_val = calculate_qnr_script_a(gen_img_np, gt_img_u8)
            metrics["qnr"].append(qnr_val)
            metrics["d_lambda"].append(dl_val)
            metrics["d_s"].append(ds_val)

            # Save images for FID
            gen_img.save(os.path.join(fid_fake_path, f"{idx_str}.png"))
            gt_pil.save(os.path.join(fid_real_path, f"{idx_str}.png"))

    # Print results
    print(f"\n{'=' * 50}")
    print(f" Evaluation Results ({cfg.data.dataset.upper()})")
    print(f"{'=' * 50}")
    print(f"Evaluated pairs: {len(metrics['psnr'])}")
    print(f"PSNR:       {np.mean(metrics['psnr']):.4f} dB")
    print(f"SSIM:       {np.mean(metrics['ssim']):.4f}")
    print(f"RMSE:       {np.mean(metrics['rmse']):.4f}")
    print(f"LPIPS:      {np.mean(metrics['lpips']):.4f}")
    print(f"DISTS:      {np.mean(metrics['dists']):.4f}")
    print(f"SCC:        {np.mean(metrics['scc']):.4f}")
    print(f"SAM (rad):  {np.mean(metrics['sam_angle']):.4f}")
    print("-" * 30)
    print(f"[Modified QNR]")
    print(f"QNR:        {np.mean(metrics['qnr']):.4f}")
    print(f"D_lambda:   {np.mean(metrics['d_lambda']):.4f}")
    print(f"D_s:        {np.mean(metrics['d_s']):.4f}")

    # FID calculation
    print("\nCalculating FID/KID/ISC...")
    try:
        fid_metrics = calculate_fid_metrics(
            input1=fid_real_path,
            input2=fid_fake_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            prc=True,
            verbose=False
        )

        print(f"\n{'=' * 50}")
        print(f" Generative Metrics Report")
        print(f"{'=' * 50}")
        print(f"FID:        {fid_metrics['frechet_inception_distance']:.4f}")
        print(f"KID:        {fid_metrics['kernel_inception_distance_mean']:.4f} +/- {fid_metrics['kernel_inception_distance_std']:.4f}")
        print(f"ISC:        {fid_metrics['inception_score_mean']:.4f} +/- {fid_metrics['inception_score_std']:.4f}")
        print(f"Precision:  {fid_metrics['precision']:.4f}")
        print(f"Recall:     {fid_metrics['recall']:.4f}")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"FID/KID calculation failed: {e}")

    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Dataset: {cfg.data.dataset}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Samples: {len(metrics['psnr'])}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {np.mean(v):.4f}\n")
    print(f"\nMetrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
