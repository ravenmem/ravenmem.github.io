"""
Visualization utilities for image tensor conversion and grid creation.
"""

import numpy as np
from PIL import Image
import torch


def tensor_to_pil(tensor):
    """
    Convert a [-1, 1] or [0, 1] tensor to PIL Image.

    Args:
        tensor: Torch tensor of shape (C, H, W)

    Returns:
        PIL.Image: RGB image
    """
    img = tensor.clone().detach().cpu()

    # If [-1, 1] range, convert to [0, 1]
    if img.min() < 0:
        img = (img / 2.0 + 0.5).clamp(0, 1)
    else:
        img = img.clamp(0, 1)

    img = img.permute(1, 2, 0).float().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return Image.fromarray(img)


def image_grid(imgs, rows, cols):
    """
    Create a grid of images.

    Args:
        imgs: List of PIL Images
        rows: Number of rows
        cols: Number of columns

    Returns:
        PIL.Image: Grid image
    """
    assert len(imgs) == rows * cols, f"Expected {rows * cols} images, got {len(imgs)}"

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
