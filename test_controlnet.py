"""
Unified evaluation script for SAR-to-Optical image synthesis.

Supports both BENv2 and SEN12MS datasets with configurable parameters.

Usage:
    # BENv2 evaluation
    python test_seesr.py --dataset benv2 --checkpoint_dir ./checkpoints/benv2/checkpoint-100000 ...

    # SEN12MS evaluation
    python test_seesr.py --dataset sen12ms --checkpoint_dir ./checkpoints/sen12ms/checkpoint-100000 ...
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, DDPMScheduler
from torchvision.transforms import v2
from safetensors.torch import load_file
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torch_fidelity import calculate_metrics as calculate_fid_metrics
import lpips
from DISTS_pytorch import DISTS
from peft import LoraConfig, get_peft_model

from configilm.extra.DataSets import BENv2_DataSet
from dataloaders.sen12ms_dataloader import SEN12MSDataset
from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from utils.transforms import make_transform
from utils.visualization import tensor_to_pil
from utils.models import ViTKDDistillationModel
from utils.prompts import (
    SD_V2_CLASS_PROMPTS, LCCS_LU_CLASS_PROMPTS_V2,
    logits_to_prompt, metadata_normalize
)
from utils.metrics import (
    calculate_qnr_script_a, calculate_sam_metric,
    calculate_rmse, calculate_scc
)
from utils.dataloaders import ben_collate_fn, sen12ms_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAR-to-Optical image synthesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset selection
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--dataset", type=str, choices=["benv2", "sen12ms"], default="benv2",
        help="Dataset to evaluate on")
    dataset_group.add_argument("--sen12ms_root", type=str, default="./sen12ms",
        help="Root directory for SEN12MS dataset")
    dataset_group.add_argument("--images_lmdb", type=str, default="/root/hyun/rico-hdl/Encoded-BigEarthNet",
        help="Path to BENv2 images LMDB")
    dataset_group.add_argument("--metadata_parquet", type=str, default="/root/hyun/meta/metadata.parquet",
        help="Path to BENv2 metadata parquet")
    dataset_group.add_argument("--metadata_snow_cloud_parquet", type=str,
        default="/root/hyun/meta/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        help="Path to BENv2 snow/cloud metadata")

    # Model paths
    model_group = parser.add_argument_group("Model Paths")
    model_group.add_argument("--checkpoint_dir", type=str, required=True,
        help="Path to trained model checkpoint directory")
    model_group.add_argument("--base_model_path", type=str,
        default="./stable-diffusion-2-1-base/stable-diffusion-2-1-base",
        help="Path to Stable Diffusion 2.1 base model")
    model_group.add_argument("--dino_repo", type=str, default="/root/hyun/dinov3",
        help="Path to DINOv3 repository")
    model_group.add_argument("--dino_weights", type=str,
        default="/root/hyun/현서/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        help="Path to DINOv3 pretrained weights")
    model_group.add_argument("--dino_checkpoint", type=str, required=True,
        help="Path to DINOv3 classifier checkpoint")

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--output_dir", type=str, default="./validation_results",
        help="Output directory for results")
    eval_group.add_argument("--batch_size", type=int, default=20,
        help="Evaluation batch size")
    eval_group.add_argument("--num_samples", type=int, default=1000,
        help="Number of samples to evaluate")
    eval_group.add_argument("--seed", type=int, default=42,
        help="Random seed")
    eval_group.add_argument("--external_folder", type=str, default=None,
        help="Path to external generated images (skip generation if provided)")
    eval_group.add_argument("--fixed_prompt", type=str, default=None,
        help="Path to external generated images (skip generation if provided)")

    # Inference settings
    inf_group = parser.add_argument_group("Inference Settings")
    inf_group.add_argument("--num_inference_steps", type=int, default=50,
        help="Number of diffusion inference steps")
    inf_group.add_argument("--guidance_scale", type=float, default=5.5,
        help="Classifier-free guidance scale")
    inf_group.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16", "f32"], default="f32",
        help="Mixed precision mode")

    # DINOv3 settings
    dino_group = parser.add_argument_group("DINOv3 Settings")
    dino_group.add_argument("--layers_to_distill", nargs='+', type=int, default=[14, 17, 20, 23],
        help="DINOv3 layers to extract features from")
    dino_group.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    dino_group.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    dino_group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    return parser.parse_args()


def load_weights_into_model(model, weight_path, model_name):
    """Load weights from safetensors file into model."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    print(f"Loading {model_name} from {os.path.basename(weight_path)}...")
    state_dict = load_file(weight_path)
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


def main():
    args = parse_args()

    # Dataset-specific configuration
    if args.dataset == "sen12ms":
        NUM_CLASSES = 11
        CLASS_PROMPTS = LCCS_LU_CLASS_PROMPTS_V2
        print(f"Using SEN12MS dataset configuration: {NUM_CLASSES} classes")
    else:  # benv2
        NUM_CLASSES = 19
        CLASS_PROMPTS = SD_V2_CLASS_PROMPTS
        print(f"Using BENv2 dataset configuration: {NUM_CLASSES} classes")

    # Device and precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Initialize metric models
    print("Initializing metric models...")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()

    dists_loss_fn = DISTS().to(device)
    dists_loss_fn.eval()

    # Load DINOv3 classifier
    print(f"Loading DINOv3 from {args.dino_repo}...")
    backbone = torch.hub.load(args.dino_repo, 'dinov3_vitl16', source='local', weights=args.dino_weights)

    classifier_model = ViTKDDistillationModel(
        backbone, num_classes=NUM_CLASSES, layers=args.layers_to_distill
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["attn.qkv", "attn.proj"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    classifier_model = get_peft_model(classifier_model, lora_config)

    checkpoint = torch.load(args.dino_checkpoint, map_location='cpu')
    classifier_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    classifier_model.to(device, dtype=weight_dtype)
    classifier_model.eval()

    # Load pipeline (if not using external images)
    if args.external_folder is None:
        print(f"Loading pipeline from {args.checkpoint_dir}...")
        scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
        vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae")
        feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.base_model_path}/feature_extractor")

        unet = UNet2DConditionModel.from_pretrained(
            args.base_model_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            use_image_cross_attention=True,
            image_encoder_hidden_states=1024
        )
        controlnet = ControlNetModel.from_unet(
            unet,
            use_image_cross_attention=True,
            image_cross_attention_dim=1024
        )

        controlnet_path = os.path.join(args.checkpoint_dir, "model.safetensors")
        unet_path = os.path.join(args.checkpoint_dir, "model_1.safetensors")

        load_weights_into_model(controlnet, controlnet_path, "ControlNet")
        load_weights_into_model(unet, unet_path, "UNet")

        pipeline = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            feature_extractor=feature_extractor, unet=unet, controlnet=controlnet,
            scheduler=scheduler, safety_checker=None, requires_safety_checker=False
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae.to(dtype=torch.float32)
        if torch.cuda.is_available():
            pipeline.enable_xformers_memory_efficient_attention()
    else:
        pipeline = None
        print(f"External evaluation mode: Loading images from {args.external_folder}")

    # Dataset setup
    RESIZE_SIZE = 256
    transform_val = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False,
                             calc_norm=True, train_datatype="opt",
                             dataset=args.dataset if args.dataset == "sen12ms" else "benv2"),
        "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=False,
                             calc_norm=True, train_datatype="sar",
                             dataset=args.dataset if args.dataset == "sen12ms" else "benv2")
    }

    if args.dataset == "sen12ms":
        val_dataset = SEN12MSDataset(
            root_dir=args.sen12ms_root,
            subset="test",
            seed=args.seed,
            transform=transform_val
        )
        collate_fn = sen12ms_collate_fn
    else:  # benv2
        datapath = {
            "images_lmdb": args.images_lmdb,
            "metadata_parquet": args.metadata_parquet,
            "metadata_snow_cloud_parquet": args.metadata_snow_cloud_parquet,
        }
        val_dataset = BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=(12, 120, 120),
            split='test',
            transform=transform_val,
            merge_patch=True,
            return_diffsat_metadata=True
        )
        collate_fn = ben_collate_fn

    # Create subset
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(val_dataset), generator=generator)[:args.num_samples]
    val_subset = Subset(val_dataset, indices)

    validation_dataloader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Output folders
    fid_real_path = os.path.join(args.output_dir, "fid_real")
    fid_fake_path = os.path.join(args.output_dir, "fid_fake")
    label_save_path = os.path.join(args.output_dir, "labels")

    os.makedirs(fid_real_path, exist_ok=True)
    os.makedirs(fid_fake_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)

    # Normalization for classifier
    normalize_tf = v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))

    # Metrics containers
    metrics = {k: [] for k in ["psnr", "ssim", "lpips", "rmse", "sam_angle", "scc", "dists", "qnr", "d_lambda", "d_s"]}

    print(f"Starting evaluation on {args.dataset.upper()} dataset...")
    generator_inf = torch.Generator(device=device).manual_seed(args.seed)

    for step, batch_data in enumerate(tqdm(validation_dataloader)):
        # Unpack batch (both datasets now return 4 items via collate functions)
        batch, lbl, md_norm, seasons = batch_data

        sar_image = batch["sar"].to(device, dtype=weight_dtype)
        opt_image = batch["opt"].to(device, dtype=weight_dtype)

        # Image loading/generation logic
        generated_images = []
        skip_generation = False

        check_dir = args.external_folder if args.external_folder else fid_fake_path
        all_exist = all([
            os.path.exists(os.path.join(check_dir, f"{(step * args.batch_size + i):05d}.png"))
            for i in range(len(sar_image))
        ])

        if all_exist:
            skip_generation = True
            for i in range(len(sar_image)):
                idx_str = f"{(step * args.batch_size + i):05d}"
                img = Image.open(os.path.join(check_dir, f"{idx_str}.png")).convert("RGB")
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.BICUBIC)
                generated_images.append(img)
        elif args.external_folder:
            for i in range(len(sar_image)):
                generated_images.append(Image.new("RGB", (256, 256), (0, 0, 0)))
        else:
            # Generate images
            sar_classifier_cond = normalize_tf(sar_image)
            with torch.no_grad():
                logits, visual_features_dict = classifier_model(sar_classifier_cond)
                feature_stack = []
                for idx in sorted(visual_features_dict.keys()):
                    feat = visual_features_dict[idx]["patch"].flatten(2).transpose(1, 2)
                    cls_token = visual_features_dict[idx]["cls"].unsqueeze(1)
                    feature_stack.append(torch.cat((cls_token, feat), dim=1))
                image_encoder_hidden_states = torch.stack(feature_stack, dim=0).to(dtype=weight_dtype)

                prompts = logits_to_prompt(
                    args=args,
                    is_train=False,
                    logits=logits,
                    class_names=CLASS_PROMPTS,
                    seasons=seasons,
                    threshold=0.7,
                    max_classes=3
                )

                output = pipeline(
                    prompts, sar_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator_inf,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=["low quality"] * len(prompts),
                    ram_encoder_hidden_states=image_encoder_hidden_states,
                    output_type="pil"
                )
                generated_images = output.images

        # Calculate metrics
        for i, gen_img in enumerate(generated_images):
            idx_str = f"{step * args.batch_size + i:05d}"

            gt_pil = tensor_to_pil(opt_image[i])
            gt_img_u8 = np.array(gt_pil)
            gen_img_np = np.array(gen_img)

            # Save labels
            if lbl is not None:
                current_label = lbl[i]
                if isinstance(current_label, torch.Tensor):
                    current_label = current_label.cpu().numpy()
                np.save(os.path.join(label_save_path, f"{idx_str}.npy"), current_label)

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

            # Save images
            if not skip_generation and args.external_folder is None:
                gen_img.save(os.path.join(fid_fake_path, f"{idx_str}.png"))
            gt_pil.save(os.path.join(fid_real_path, f"{idx_str}.png"))

    # Final report
    print(f"\n{'=' * 50}")
    print(f" Evaluation Results ({args.dataset.upper()})")
    print(f"{'=' * 50}")
    print(f"Evaluated pairs: {len(metrics['psnr'])}")
    print(f"PSNR:       {np.mean(metrics['psnr']):.4f} dB")
    print(f"SSIM:       {np.mean(metrics['ssim']):.4f}")
    print(f"RMSE:       {np.mean(metrics['rmse']):.4f}")
    print(f"LPIPS:      {np.mean(metrics['lpips']):.4f}")
    print(f"DISTS:      {np.mean(metrics['dists']):.4f}")
    print(f"SCC:        {np.mean(metrics['scc']):.4f}")
    print(f"SAM (rad):  {np.mean(metrics['sam_angle']):.4f} (Lower is better)")
    print("-" * 30)
    print(f"[Modified QNR]")
    print(f"QNR:        {np.mean(metrics['qnr']):.4f}")
    print(f"D_lambda:   {np.mean(metrics['d_lambda']):.4f}")
    print(f"D_s:        {np.mean(metrics['d_s']):.4f}")

    # FID calculation
    print("\nCalculating FID/KID/ISC...")
    fid_input2 = args.external_folder if args.external_folder else fid_fake_path
    try:
        fid_metrics = calculate_fid_metrics(
            input1=fid_real_path,
            input2=fid_input2,
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


if __name__ == "__main__":
    main()
