"""
Model utilities and wrapper classes for DINOv3 and diffusion models.
"""

import logging
import torch
import torch.nn as nn
from typing import List
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class ViTKDDistillationModel(nn.Module):
    """
    DINOv3 wrapper for multi-layer feature extraction and classification.
    Returns both classification logits and intermediate feature maps.
    """

    def __init__(self, backbone, num_classes: int, layers: List[int]):
        """
        Args:
            backbone: DINOv3 backbone model
            num_classes: Number of output classes
            layers: List of layer indices to extract features from
        """
        super().__init__()
        self.backbone = backbone
        d = self.backbone.embed_dim

        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
        self.layers_to_extract = sorted(list(set(layers)))
        self.output_map = {layer_idx: i for i, layer_idx in enumerate(self.layers_to_extract)}

        if not self.layers_to_extract:
            raise ValueError("layers list cannot be empty.")
        self.final_cls_layer_idx = max(self.layers_to_extract)

    def forward(self, pixel_values):
        intermediate_outputs = self.backbone.get_intermediate_layers(
            pixel_values,
            n=self.layers_to_extract,
            reshape=False,
            norm=True,
            return_class_token=True
        )

        input_h, input_w = pixel_values.shape[-2:]
        patch_size = self.backbone.patch_size
        H = input_h // patch_size
        W = input_w // patch_size

        intermediate_features = {}
        for layer_idx in self.layers_to_extract:
            output_idx = self.output_map[layer_idx]
            patch_tokens, cls_token = intermediate_outputs[output_idx]
            B, N, C = patch_tokens.shape
            patch_map = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

            intermediate_features[layer_idx] = {
                "patch": patch_map,
                "cls": cls_token
            }

        final_idx = self.output_map[max(self.layers_to_extract)]
        _, cls_tokens_final = intermediate_outputs[final_idx]
        logits = self.head(cls_tokens_final)

        return logits, intermediate_features


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """
    Dynamically import the appropriate text encoder class.

    Args:
        pretrained_model_name_or_path: HuggingFace model path
        revision: Model revision

    Returns:
        Text encoder class
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def zero_init_image_attentions(model, model_name):
    """
    Zero-initialize image attention output projection layers.

    This ensures that at the start of training, image cross-attention
    has no effect, allowing gradual learning.

    Args:
        model: UNet or ControlNet model
        model_name: Name for logging

    Returns:
        List of initialized layer names
    """
    init_count = 0
    target_layer_names = []

    for name, module in model.named_modules():
        if 'image_attentions' in name and name.endswith('proj_out'):
            target_layer_names.append(name)

            if hasattr(module, 'weight'):
                torch.nn.init.zeros_(module.weight)
                init_count += 1

            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    if init_count > 0:
        logger.info(f"Initialized {init_count} output projection layers in {model_name} to zeros.")
    else:
        logger.warning(f"Could not find any layers to initialize in {model_name}.")

    return target_layer_names
