"""
Diffusion model loss functions for ControlNet training.

This module contains loss functions specific to diffusion-based
SAR-to-Optical image synthesis.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedDiffusionLoss(nn.Module):
    """
    Confidence-weighted diffusion loss for uncertainty-aware training.

    The model predicts both noise and a confidence map. The loss is weighted
    by the confidence, encouraging the model to be uncertain where predictions
    are difficult.

    Args:
        loss_beta: Beta parameter for confidence weighting (higher = more influence)
        lpips_weight: Weight for LPIPS perceptual loss (0 to disable)
    """

    def __init__(
        self,
        loss_beta: float = 1.0,
        lpips_weight: float = 0.0,
    ):
        super().__init__()
        self.loss_beta = loss_beta
        self.lpips_weight = lpips_weight
        self.tau = math.sqrt(math.log(2 * math.pi))

        # Initialize LPIPS if needed
        if lpips_weight > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex')
                self.lpips_fn.requires_grad_(False)
            except ImportError:
                raise ImportError("LPIPS loss requires: pip install lpips")
        else:
            self.lpips_fn = None

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        decoded_pred: Optional[torch.Tensor] = None,
        decoded_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence-weighted diffusion loss.

        Args:
            model_pred: Model's noise prediction
            target: Target noise or velocity
            confidence: Confidence map from model
            decoded_pred: Optional decoded prediction for LPIPS
            decoded_target: Optional decoded target for LPIPS

        Returns:
            Dictionary containing:
                - loss: Total loss
                - loss_recon: Reconstruction (MSE) loss
                - loss_reg: Regularization loss
                - loss_lpips: LPIPS loss (if enabled)
        """
        # Confidence-weighted reconstruction loss
        weighted_residual = (target - model_pred) * (confidence ** self.loss_beta)
        loss_recon = (weighted_residual ** 2).mean()

        # Regularization loss (encourages non-zero confidence)
        loss_reg = -torch.log(confidence ** self.loss_beta + 1e-8).mean() + (self.tau ** 2) / 2.0

        total_loss = loss_recon + loss_reg

        result = {
            "loss": total_loss,
            "loss_recon": loss_recon,
            "loss_reg": loss_reg,
        }

        # Optional LPIPS loss
        if self.lpips_fn is not None and decoded_pred is not None and decoded_target is not None:
            # LPIPS expects [-1, 1] range
            with torch.no_grad():
                loss_lpips = self.lpips_fn(decoded_pred, decoded_target).mean()
            result["loss_lpips"] = loss_lpips
            result["loss"] = total_loss + self.lpips_weight * loss_lpips

        return result


class SimpleDiffusionLoss(nn.Module):
    """
    Simple MSE diffusion loss without confidence weighting.

    This is the standard diffusion training objective.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute simple MSE loss between prediction and target.

        Args:
            model_pred: Model's noise prediction
            target: Target noise or velocity

        Returns:
            MSE loss value
        """
        return F.mse_loss(model_pred, target, reduction=self.reduction)


class SNRWeightedDiffusionLoss(nn.Module):
    """
    Signal-to-Noise Ratio weighted diffusion loss.

    Weights the loss based on the SNR at each timestep, following the
    Min-SNR weighting strategy from 'Efficient Diffusion Training via
    Min-SNR Weighting Strategy'.

    Args:
        snr_gamma: Gamma parameter for Min-SNR weighting (default: 5.0)
    """

    def __init__(self, snr_gamma: float = 5.0):
        super().__init__()
        self.snr_gamma = snr_gamma

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SNR-weighted diffusion loss.

        Args:
            model_pred: Model's noise prediction
            target: Target noise
            timesteps: Diffusion timesteps
            alphas_cumprod: Cumulative product of alphas from noise scheduler

        Returns:
            SNR-weighted MSE loss
        """
        # Compute SNR for each timestep
        snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])

        # Min-SNR weighting
        snr_weight = torch.clamp(snr, max=self.snr_gamma) / snr

        # Compute per-sample loss
        mse_loss = F.mse_loss(model_pred, target, reduction="none")
        mse_loss = mse_loss.mean(dim=list(range(1, mse_loss.ndim)))

        # Apply SNR weighting
        weighted_loss = (snr_weight * mse_loss).mean()

        return weighted_loss


def compute_snr(timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio for given timesteps.

    Args:
        timesteps: Diffusion timesteps (B,)
        alphas_cumprod: Cumulative product of alphas

    Returns:
        SNR values for each timestep (B,)
    """
    alpha = alphas_cumprod[timesteps]
    return alpha / (1 - alpha)
