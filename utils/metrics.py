"""
Remote sensing image quality metrics for SAR-to-Optical evaluation.

This module provides metrics for evaluating synthesized optical images:
- QNR (Quality with No Reference)
- SAM (Spectral Angle Mapper)
- SCC (Spatial Correlation Coefficient)
- RMSE (Root Mean Square Error)
"""

import numpy as np
import cv2


def _to_float01(img):
    """Convert image to float64 in [0, 1] range."""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _uiqi(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Universal Image Quality Index over whole image (Global).

    Args:
        x: First image array (flattened internally)
        y: Second image array (flattened internally)
        eps: Small constant for numerical stability

    Returns:
        UIQI score (float)
    """
    x = x.astype(np.float64).flatten()
    y = y.astype(np.float64).flatten()
    mx = x.mean()
    my = y.mean()
    vx = x.var()
    vy = y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = (vx + vy) * (mx * mx + my * my) + eps
    num = 4.0 * cov * mx * my
    q = num / denom
    if np.isnan(q) or np.isinf(q):
        return 0.0
    return float(q)


def _rgb_to_luminance(img01: np.ndarray) -> np.ndarray:
    """Convert RGB image to luminance using standard coefficients."""
    if img01.ndim == 2:
        return img01
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _highpass(img01_gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian high-pass filter: HP = I - GaussianBlur(I)."""
    blur = cv2.GaussianBlur(img01_gray, (5, 5), 2.0)
    hp = img01_gray - blur
    return hp


def calculate_qnr_script_a(gen_img: np.ndarray, gt_img: np.ndarray,
                           alpha: float = 1.0, beta: float = 1.0):
    """
    Calculate QNR (Quality with No Reference) metric.

    This implementation compares generated image directly to ground truth
    using high-pass filtering for spatial quality assessment.

    Args:
        gen_img: Generated image (H, W, C), uint8 [0-255] or float [0-1]
        gt_img: Ground truth image (H, W, C), uint8 [0-255] or float [0-1]
        alpha: Weight for spectral distortion term
        beta: Weight for spatial distortion term

    Returns:
        Tuple of (QNR, D_lambda, D_s)
    """
    gen = _to_float01(gen_img)
    gt = _to_float01(gt_img)

    if gen.ndim == 2:
        gen = np.stack([gen, gen, gen], axis=2)
    if gt.ndim == 2:
        gt = np.stack([gt, gt, gt], axis=2)

    # D_lambda: Spectral distortion
    c = gen.shape[2]
    if c < 2:
        d_lambda = 0.0
    else:
        diffs = []
        for i in range(c):
            for j in range(i + 1, c):
                q_gen = _uiqi(gen[..., i], gen[..., j])
                q_gt = _uiqi(gt[..., i], gt[..., j])
                diffs.append(abs(q_gen - q_gt))
        d_lambda = float(np.mean(diffs)) if diffs else 0.0

    # D_s: Spatial distortion using high-pass UIQI
    pan = _rgb_to_luminance(gt)
    pan_hp = _highpass(pan)
    ds_terms = []
    for k in range(c):
        gen_hp = _highpass(gen[..., k])
        gt_hp = _highpass(gt[..., k])
        q_gen_pan = _uiqi(gen_hp, pan_hp)
        q_gt_pan = _uiqi(gt_hp, pan_hp)
        ds_terms.append(abs(q_gen_pan - q_gt_pan))
    d_s = float(np.mean(ds_terms)) if ds_terms else 0.0

    qnr = (1.0 - d_lambda) ** alpha * (1.0 - d_s) ** beta
    return float(qnr), d_lambda, d_s


def calculate_sam_metric(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Spectral Angle Mapper (SAM).

    Measures spectral similarity between two multi-band images by computing
    the angle between their spectral vectors at each pixel.

    Args:
        img1: First image (H, W, C), typically ground truth
        img2: Second image (H, W, C), typically generated

    Returns:
        Mean spectral angle in radians (lower is better, 0 = identical)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Dot product along channel axis
    dot_product = np.sum(img1 * img2, axis=2)

    # Norms
    norm_img1 = np.linalg.norm(img1, axis=2)
    norm_img2 = np.linalg.norm(img2, axis=2)

    # Denominator (product of norms)
    denom = norm_img1 * norm_img2
    denom[denom == 0] = 1e-6  # Avoid division by zero

    # Cosine theta
    cos_theta = dot_product / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # ArcCos to get angles (radians)
    angles = np.arccos(cos_theta)

    # Mean SAM
    mean_sam = np.mean(angles)

    if np.isnan(mean_sam):
        return 0.0

    return float(mean_sam)


def calculate_scc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Spatial Correlation Coefficient (SCC).

    Based on Zhou et al. (1998), this metric computes the correlation
    between high-frequency components (spatial details) of two images
    using a Laplacian filter.

    Args:
        img1: First image (H, W, C) or (H, W)
        img2: Second image (H, W, C) or (H, W)

    Returns:
        SCC value (higher is better, 1.0 = perfect correlation)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Laplacian kernel (high-pass filter)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Apply high-pass filter
    if img1.ndim == 3:
        h, w, c = img1.shape
        img1_hp = np.zeros_like(img1)
        img2_hp = np.zeros_like(img2)
        for i in range(c):
            img1_hp[:, :, i] = cv2.filter2D(img1[:, :, i], -1, kernel)
            img2_hp[:, :, i] = cv2.filter2D(img2[:, :, i], -1, kernel)
    else:
        img1_hp = cv2.filter2D(img1, -1, kernel)
        img2_hp = cv2.filter2D(img2, -1, kernel)

    # Flatten for correlation computation
    vec1 = img1_hp.flatten()
    vec2 = img2_hp.flatten()

    # Check for flat images
    if np.std(vec1) < 1e-6 or np.std(vec2) < 1e-6:
        return 0.0

    # Pearson correlation coefficient
    scc = np.corrcoef(vec1, vec2)[0, 1]

    if np.isnan(scc):
        return 0.0

    return float(scc)


def calculate_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        img1: First image
        img2: Second image

    Returns:
        RMSE value (lower is better)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float(np.sqrt(mse))
