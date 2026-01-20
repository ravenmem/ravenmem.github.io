import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import gc
import warnings

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import CLIPTextModel, CLIPTokenizer
from configilm.extra.DataSets import BENv2_DataSet

from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from peft import LoraConfig, get_peft_model

# Utils imports
from utils.transforms import make_transform
from utils.visualization import tensor_to_pil
from utils.prompts import (
    BEN_V2_CLASS_NAMES, SEASON_MAP, SD_V2_CLASS_PROMPTS, LCCS_LU_CLASS_PROMPTS_V2,
    get_season_from_month, metadata_normalize, logits_to_prompt
)
from utils.dataloaders import ben_collate_fn, sen12ms_collate_fn
from utils.models import (
    ViTKDDistillationModel,
    import_model_class_from_model_name_or_path,
    zero_init_image_attentions
)
from utils.pipeline import load_seesr_pipeline, save_model_card
from utils.validation import run_validation

# SEN12MS imports
from torch.utils.data.distributed import DistributedSampler
from dataloaders.sen12ms_dataloader import SEN12MSDataset


if is_wandb_available():
    import wandb

check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["benv2", "sen12ms"], default="benv2",
        help="Dataset to use: 'benv2' for BigEarthNet v2, 'sen12ms' for SEN12MS")
    parser.add_argument("--sen12ms_root", type=str, default="./sen12ms",
        help="Root directory for SEN12MS dataset")
    parser.add_argument("--sen12ms_dino_checkpoint", type=str,
        default="/mnt/e/checkpoint_sen12ms/stage1_sar/checkpoint_stage1_epoch293.pth",
        help="DINOv3 checkpoint trained on SEN12MS (11 classes)")
    parser.add_argument("--sen12_root", type=str, default="/mnt/f/sen12_split", help="Root directory for SEN12 dataset")
    parser.add_argument("--lpips_weight", type=float, default=0.1, help="Weight of LPIPS loss")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-base",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experience/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='NOTHING',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SeeSR",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["image_attentions"])



    parser.add_argument("--warmup_steps", type=int, default=5000, help="Number of warmup steps for C-Diff style training before applying wavelet loss.")
    parser.add_argument("--wavelet_alpha", type=float, default=10.0, help="Alpha scaling factor for wavelet mismatch. Higher value means stronger penalty for mismatch.")
    parser.add_argument("--c_diff_beta", type=int, default=1, help="Beta value for the C-Diff loss, as in β-NLL.")
    parser.add_argument(
        "--use_diffsat", action="store_true", help="use diffsat"
    )

    # DINOv3 Configuration
    parser.add_argument("--dino_repo_path", type=str, default="/root/hyun/dinov3", help="Path to DINOv3 repository")
    parser.add_argument("--dino_checkpoint", type=str, default="/root/hyun/현서/checkpoints/checkpoint_stage1_epoch68.pth", help="Path to DINOv3 classifier checkpoint")
    parser.add_argument("--dino_weights", type=str, default="/root/hyun/현서/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", help="Path to DINOv3 backbone pretrained weights")
    parser.add_argument("--num_classes", type=int, default=19, help="Number of classes for classifier")
    parser.add_argument("--merge_patch", action="store_true", default=True, help="Whether to merge patches for higher resolution")
    parser.add_argument("--no_merge_patch", action="store_false", dest="merge_patch", help="Disable patch merging")
    parser.add_argument("--layers_to_distill", nargs='+', type=int, default=[14, 17, 20, 23], help="DINOv3 layers to extract features from")

    # LoRA Configuration
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Dataset Paths
    parser.add_argument("--images_lmdb", type=str, default="/root/hyun/rico-hdl/Encoded-BigEarthNet", help="Path to images LMDB")
    parser.add_argument("--metadata_parquet", type=str, default="/root/hyun/meta/metadata.parquet", help="Path to metadata parquet")
    parser.add_argument("--metadata_snow_cloud_parquet", type=str, default="/root/hyun/meta/metadata_for_patches_with_snow_cloud_or_shadow.parquet", help="Path to snow/cloud metadata parquet")

    # Validation/Inference Parameters
    parser.add_argument("--validation_inference_steps", type=int, default=100, help="Number of inference steps during validation")
    parser.add_argument("--validation_guidance_scale", type=float, default=5.5, help="Guidance scale during validation")
    parser.add_argument("--validation_max_batches", type=int, default=5, help="Maximum number of batches during validation")
    parser.add_argument("--validation_seed", type=int, default=500, help="Random seed for validation")

    # Prompt Configuration
    parser.add_argument("--prompt_threshold", type=float, default=0.7, help="Threshold for class probability in prompt generation")
    parser.add_argument("--prompt_max_classes", type=int, default=2, help="Maximum number of classes in prompt")
    parser.add_argument("--fixed_prompt", type=str, default="Electro-Optical Image", help="Fixed prompt override (empty string to disable)")

    # Loss Configuration
    parser.add_argument("--loss_beta", type=float, default=1.0, help="Beta for uncertainty weighting in loss")
    parser.add_argument("--metadata_dropout", type=float, default=0.1, help="Metadata dropout probability")

    # Negative prompt for validation
    parser.add_argument("--negative_prompt", type=str,
        default="low quality, worst quality, blurry, noisy, jpeg artifacts, speckle, speckle noise, grainy, monochrome, grayscale, dark",
        help="Negative prompt for validation inference")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[ddp_kwargs]
)


# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=True)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)


################# Dataset-specific Configuration #################

RESIZE_SIZE = 256 if args.merge_patch else 128

# Set dataset-specific number of classes and prompts
if args.dataset == "sen12ms":
    NUM_CLASSES = 11
    CLASS_PROMPTS = LCCS_LU_CLASS_PROMPTS_V2
    DINO_CHECKPOINT = args.sen12ms_dino_checkpoint
    logger.info(f"Using SEN12MS dataset configuration: {NUM_CLASSES} classes")
else:  # benv2
    NUM_CLASSES = args.num_classes  # default 19
    CLASS_PROMPTS = SD_V2_CLASS_PROMPTS
    DINO_CHECKPOINT = args.dino_checkpoint
    logger.info(f"Using BENv2 dataset configuration: {NUM_CLASSES} classes")

################# DINOv3 Linear Classifier with LoRA #################

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    backbone = torch.hub.load(args.dino_repo_path, 'dinov3_vitl16', source='local', weights=args.dino_weights)

dino_hidden_dim = backbone.embed_dim
classifier_model = ViTKDDistillationModel(backbone, num_classes=NUM_CLASSES, layers=args.layers_to_distill)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["attn.qkv", "attn.proj"],
    lora_dropout=args.lora_dropout,
    bias="none"
)
classifier_model = get_peft_model(classifier_model, lora_config)
checkpoint = torch.load(DINO_CHECKPOINT, map_location='cpu')
classifier_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

classifier_model.to(accelerator.device)
classifier_model.requires_grad_(False)
classifier_model.eval()
################# DINOv3 Linear Classifier with LoRA #################

if args.unet_model_name_or_path:
    logger.info("Loading unet weights from self-train")
    unet = UNet2DConditionModel.from_pretrained_orig(
        pretrained_model_path=args.pretrained_model_name_or_path,
        seesr_model_path=args.unet_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        use_image_cross_attention=True,
        image_cross_attention_dim=dino_hidden_dim,
    )
else:
    logger.info("Loading unet weights from SD")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        use_image_cross_attention=True,
        low_cpu_mem_usage=False,
        image_cross_attention_dim=dino_hidden_dim,            
    )

if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path, subfolder="controlnet"
    )
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(
        unet,
        use_image_cross_attention=True,
        image_cross_attention_dim=dino_hidden_dim,
    )

if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            
            model = models.pop()

            if not isinstance(model, UNet2DConditionModel):
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True

            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_load_state_pre_hook(load_model_hook)


vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

controlnet.train()
controlnet.requires_grad_(True)


for name, module in unet.named_modules():
    if name.endswith(tuple(args.trainable_modules)):
        logger.info(f'{name} in <unet> will be optimized.')
        for params in module.parameters():
            params.requires_grad = True

# Zero-initialize image attention projections
zero_init_image_attentions(unet, "UNet")
zero_init_image_attentions(controlnet, "ControlNet")

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    # vae.enable_gradient_checkpointing()

# Check that all trainable models are in full precision
low_precision_error_string = (
    " Please make sure to always have all model weights in full float32 precision when starting training - even if"
    " doing mixed precision training, copy of the weights should still be float32."
)

if accelerator.unwrap_model(controlnet).dtype != torch.float32:
    raise ValueError(
        f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    )
if accelerator.unwrap_model(unet).dtype != torch.float32:
    raise ValueError(
        f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
    )


if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

logger.info("Setting up optimizer for ControlNet and UNet")
params_to_optimize = list(controlnet.parameters()) + list(unet.parameters())
logger.info("Loading optimizer...")
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

################# Dataset and DataLoader Setup #################

if args.dataset == "sen12ms":
    # SEN12MS dataset configuration
    dataset_name = "sen12ms"
    transform = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=True, calc_norm=True, train_datatype="opt", dataset="sen12ms"),
        "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=True, calc_norm=True, train_datatype="sar", dataset="sen12ms")
    }
    transform_val = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False, calc_norm=True, train_datatype="opt", dataset="sen12ms"),
        "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=False, calc_norm=True, train_datatype="sar", dataset="sen12ms")
    }

    train_dataset = SEN12MSDataset(root_dir=args.sen12ms_root, subset="train", seed=args.seed or 42, transform=transform)
    val_dataset = SEN12MSDataset(root_dir=args.sen12ms_root, subset="test", seed=args.seed or 42, transform=transform_val)

    # Use DistributedSampler for SEN12MS
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # Sampler handles shuffling
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=sen12ms_collate_fn,
    )

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=sen12ms_collate_fn,
    )

    logger.info(f"SEN12MS dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")

else:  # benv2
    # BENv2 dataset configuration
    dataset_name = "benv2"
    datapath = {
        "images_lmdb": args.images_lmdb,
        "metadata_parquet": args.metadata_parquet,
        "metadata_snow_cloud_parquet": args.metadata_snow_cloud_parquet,
    }

    transform = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=True, calc_norm=True, train_datatype="opt"),
        "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=True, calc_norm=True, train_datatype="sar")
    }
    transform_val = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False, calc_norm=True, train_datatype="opt"),
        "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=False, calc_norm=True, train_datatype="sar")
    }

    train_dataset = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='train',
        transform=transform,
        merge_patch=args.merge_patch,
        return_diffsat_metadata=True
    )

    val_dataset = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='test',
        transform=transform_val,
        merge_patch=args.merge_patch,
        return_diffsat_metadata=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=ben_collate_fn,
    )

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ben_collate_fn,
    )

    logger.info(f"BENv2 dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")



overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    controlnet, unet, optimizer, train_dataloader, lr_scheduler
)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

classifier_model.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)
    
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

if accelerator.is_main_process:
    tracker_config = dict(vars(args))

    tracker_config.pop("validation_prompt")
    tracker_config.pop("validation_image")
    tracker_config.pop("trainable_modules")

    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


global_step = 0
first_epoch = 0
save_model_iter=1

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    disable=not accelerator.is_local_main_process,
)

from torchvision.transforms import v2

NORM_MEAN=(0.430, 0.411, 0.296)
NORM_STD=(0.213, 0.156, 0.143)

normalize = v2.Normalize(mean=NORM_MEAN, std=NORM_STD)

for epoch in range(first_epoch, args.num_train_epochs):
    for step, (batch, lbl, md_norm, seasons) in enumerate(train_dataloader):
        with accelerator.accumulate(controlnet), accelerator.accumulate(unet):
            opt_image_256 = batch["opt"].to(accelerator.device, dtype=weight_dtype)
            sar_image_256 = batch["sar"].to(accelerator.device, dtype=weight_dtype)

            sar_classifier_cond = normalize(sar_image_256)

            # Handle metadata (None for SEN12MS, tensor for BENv2)
            if md_norm is not None:
                md_norm = md_norm.to(accelerator.device)


            opt_image_vae = opt_image_256 * 2.0 - 1.0

            sar_image_cond = sar_image_256 
    

            with torch.no_grad():
                logits, visual_features_dict = classifier_model(sar_classifier_cond)

            sorted_layer_indices = sorted(visual_features_dict.keys())

            feature_stack = []
            for idx in sorted_layer_indices:
                # Patch Feature: [B, C, H, W]
                patch_tokens_map = visual_features_dict[idx]["patch"]
                cls_token = visual_features_dict[idx]["cls"]
                patch_tokens_seq = patch_tokens_map.flatten(2).transpose(1, 2)

                cls_token_seq = cls_token.unsqueeze(1)

                full_sequence = torch.cat((cls_token_seq, patch_tokens_seq), dim=1)
                
                feature_stack.append(full_sequence)            


            image_encoder_hidden_states = torch.stack(feature_stack, dim=0)
            image_encoder_hidden_states = image_encoder_hidden_states.to(accelerator.device, dtype=weight_dtype)

            prompts = logits_to_prompt(
                args=args,
                is_train=True,
                logits=logits,
                class_names=CLASS_PROMPTS,  # Uses dataset-specific prompts
                seasons=seasons,
                threshold=args.prompt_threshold,
                max_classes=args.prompt_max_classes
            )

            inputs = tokenizer(
                prompts, 
                max_length=tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )


            with torch.no_grad():
                input_ids = inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                latents = vae.encode(opt_image_vae).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if md_norm is not None:
                keep_mask = torch.rand_like(md_norm) > args.metadata_dropout
                md_norm = md_norm * keep_mask + 0. * ~keep_mask * torch.ones_like(md_norm)


            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                metadata=md_norm,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=sar_image_cond,
                return_dict=False,
                image_encoder_hidden_states=image_encoder_hidden_states,
                is_multiscale_latent=True
            )

            out = unet(
                noisy_latents,
                timesteps,
                # metadata=md_norm,
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
            conf = out[1]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            tau = math.sqrt(math.log(2 * math.pi))

            weighted_residual = (target - model_pred) * (conf ** args.loss_beta)
            loss_recon = (weighted_residual ** 2).mean()

            loss_reg = - torch.log(conf ** args.loss_beta + 1e-8).mean() + (tau**2) / 2.0

            loss = loss_recon + loss_reg

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = list(controlnet.parameters()) + list(unet.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                                      
                    logger.info(f"Saved UNet and ControlNet to {save_path}")

                if global_step % args.checkpointing_steps == 0:

                    valid_weight_dtype = torch.float32
                    torch.cuda.empty_cache()
                    gc.collect()

                    with torch.no_grad():
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(args.validation_seed)

                        pipeline = load_seesr_pipeline(
                            args,
                            accelerator,
                            True,
                            global_step,
                            unwrapped_unet,
                            unwrapped_controlnet
                        )

                        psnr_scores = []
                        fid_real_path = os.path.join(args.output_dir, "fid_real_images")
                        fid_fake_path = os.path.join(args.output_dir, "fid_fake_images")

                        if accelerator.is_main_process:
                            if os.path.exists(fid_real_path): shutil.rmtree(fid_real_path)
                            if os.path.exists(fid_fake_path): shutil.rmtree(fid_fake_path)
                            os.makedirs(fid_real_path, exist_ok=True)
                            os.makedirs(fid_fake_path, exist_ok=True)

                        for step, (batch, lbl, md_norm, seasons) in enumerate(validation_dataloader):
                            if step >= args.validation_max_batches: break
                            opt_image_256 = batch["opt"].to(accelerator.device, dtype=valid_weight_dtype)
                            sar_image_256 = batch["sar"].to(accelerator.device, dtype=valid_weight_dtype)

                            sar_classifier_cond = normalize(sar_image_256).to(accelerator.device, dtype=weight_dtype)

                            # Handle metadata (None for SEN12MS, tensor for BENv2)
                            if md_norm is not None:
                                md_norm = md_norm.to(accelerator.device)

                    
                            
                            with torch.no_grad():
                                logits, visual_features_dict = classifier_model(sar_classifier_cond)
                                sorted_layer_indices = sorted(visual_features_dict.keys())

                                feature_stack = []
                                for idx in sorted_layer_indices:
                                    feat = visual_features_dict[idx]["patch"]
                                    feat = feat.flatten(2).transpose(1, 2)
                                    feature_stack.append(feat)
                                

                            image_encoder_hidden_states = torch.stack(feature_stack, dim=0)

                            image_encoder_hidden_states = image_encoder_hidden_states.to(accelerator.device, dtype=valid_weight_dtype)
                            
                            prompts = logits_to_prompt(
                                args=args,
                                is_train=False,
                                logits=logits,
                                class_names=CLASS_PROMPTS,  # Uses dataset-specific prompts
                                seasons=seasons,
                                threshold=args.prompt_threshold,
                                max_classes=args.prompt_max_classes
                            )

                            negative_prompt = "low quality, worst quality, blurry, noisy, jpeg artifacts, speckle, speckle noise, grainy, monochrome, grayscale, dark"
                            negative_prompt_list = [negative_prompt] * len(prompts)

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


                            with torch.no_grad():
                                # opt_image_256 -> latents
                                opt_image_vae = opt_image_256 * 2.0 - 1.0
                                latents_val = vae.encode(opt_image_vae.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                                latents_val = latents_val * vae.config.scaling_factor

                                # 간단히 하나의 timestep 사용 (예: 중간 정도)
                                bsz = latents_val.shape[0]
                                t_conf = torch.randint(
                                    0,
                                    noise_scheduler.config.num_train_timesteps,
                                    (bsz,),
                                    device=latents_val.device,
                                ).long()

                                noise_val = torch.randn_like(latents_val)
                                noisy_latents_val = noise_scheduler.add_noise(latents_val, noise_val, t_conf)

                                # ControlNet로 residual 생성
                                down_block_res_val, mid_block_res_val = controlnet(
                                    noisy_latents_val,
                                    t_conf,
                                    metadata=md_norm,
                                    encoder_hidden_states=encoder_hidden_states,  # 이미 위에서 계산되어 있음
                                    controlnet_cond=sar_image_256,
                                    return_dict=False,
                                    image_encoder_hidden_states=image_encoder_hidden_states,
                                    is_multiscale_latent=True,
                                )

                                # U-Net forward 로 conf 추출
                                out_val = unet(
                                    noisy_latents_val,
                                    t_conf,
                                    # metadata=md_norm,
                                    encoder_hidden_states=encoder_hidden_states,
                                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_block_res_val],
                                    mid_block_additional_residual=mid_block_res_val.to(dtype=weight_dtype),
                                    image_encoder_hidden_states=image_encoder_hidden_states,
                                    is_multiscale_latent=True,
                                    return_dict=True
                                )
                                noise_pred_val, conf_val = out_val[0], out_val[1]
                            
                            generated_images_256 = output.images

                            for i, gen_img_256 in enumerate(generated_images_256):
                                
                                idx_str = f"{step * args.train_batch_size + i:05d}"
                                
                                gt_pil_256 = tensor_to_pil(opt_image_256[i])
                                gt_np_256 = np.array(gt_pil_256)
                                
                                cond_pil_256 = tensor_to_pil(sar_image_256[i])
                                gen_np_256 = np.array(gen_img_256)
                                
                                psnr = calculate_psnr(gt_np_256, gen_np_256, data_range=255)
                                psnr_scores.append(psnr)
                                
                                gen_img_256.save(os.path.join(fid_fake_path, f"{idx_str}.png"))
                                gt_pil_256.save(os.path.join(fid_real_path, f"{idx_str}.png"))


                                conf_map = conf_val[i, 0].detach().cpu().numpy()
                                # normalize to 0-1 for visualization
                                conf_min = conf_map.min()
                                conf_max = conf_map.max()
                                if conf_max > conf_min:
                                    conf_vis = (conf_map - conf_min) / (conf_max - conf_min)
                                else:
                                    conf_vis = np.zeros_like(conf_map)

                                # HxW -> HxW (grayscale), wandb.Image가 받아줄 수 있음
                                conf_vis_img = (conf_vis * 255).astype(np.uint8)
                                conf_pil = Image.fromarray(conf_vis_img)  # mode='L'

                                image_logs = {
                                    f"validation_samples-{idx_str}": [
                                        wandb.Image(gen_np_256, caption="Generated (256px)"),
                                        wandb.Image(np.array(cond_pil_256), caption="Input SAR (256px)"),
                                        wandb.Image(gt_np_256, caption="GT Optical (256px)"),
                                        wandb.Image(conf_pil, caption="Confidence Map"),
                                    ]
                                }
                                accelerator.log(image_logs, step=global_step)

                        avg_psnr = np.mean(psnr_scores)
                        fid_score = 0.0
                        
                        if accelerator.is_main_process:
                            metrics_dict = calculate_metrics(input1=fid_real_path, input2=fid_fake_path, cuda=True, isc=False, fid=True, kid=False, prc=False)
                            fid_score = metrics_dict['frechet_inception_distance']
                            shutil.rmtree(fid_real_path)
                            shutil.rmtree(fid_fake_path)

                        final_metrics = {"validation_avg_psnr": avg_psnr, "validation_fid": fid_score}
                        accelerator.log(final_metrics, step=global_step)
                        print(f"Validation Average PSNR: {avg_psnr:.4f} dB")
                        print(f"Validation FID: {fid_score:.4f}")
                        torch.cuda.empty_cache()

                    del pipeline
                    torch.cuda.empty_cache()

        logs = {
            "loss": loss.detach().item(),
            "loss_recon": loss_recon.detach().item(),
            "loss_reg": loss_reg.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0]
        }
        
        progress_bar.set_postfix(**logs)
        
        if global_step % 1 == 0:
            accelerator.log(logs, step=global_step)

            del loss
            del latents 
            del encoder_hidden_states 
            del image_encoder_hidden_states

            del logits
            del visual_features_dict
            del feature_stack
            gc.collect()

        if global_step >= args.max_train_steps:
            break

# Create the pipeline using using the trained modules and save it.
if accelerator.is_main_process:
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(args.output_dir)

    unet = accelerator.unwrap_model(unet)
    unet.save_pretrained(args.output_dir)

    if args.push_to_hub:
        save_model_card(
            repo_id,
            image_logs=image_logs,
            base_model=args.pretrained_model_name_or_path,
            repo_folder=args.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

accelerator.end_training()
