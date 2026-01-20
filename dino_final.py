"""
DINOv3 Knowledge Distillation Training for SAR-to-Optical Translation

This script implements a 2-stage training process for fine-tuning DINOv3 with LoRA
on BigEarthNet-v2 dataset for land cover classification.

2-Stage Training Process:
    Stage 0 (Optical Baseline):
        Train DINOv3 with LoRA on optical images to create a teacher model.

        Example:
            CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \\
                --stage 0 \\
                --data_type opt \\
                --output_dir ./checkpoints/stage0_opt \\
                --batch_size 72 \\
                --num_epochs 100 \\
                --learning_rate_base 1e-4

    Stage 1 (SAR with Knowledge Distillation):
        Train DINOv3 with LoRA on SAR images using knowledge distillation from
        the optical teacher model created in Stage 0.

        Example:
            CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \\
                --stage 1 \\
                --data_type sar \\
                --teacher_checkpoint ./checkpoints/stage0_opt/checkpoint_stage0_epoch100.pth \\
                --output_dir ./checkpoints/stage1_sar \\
                --batch_size 72 \\
                --num_epochs 100 \\
                --learning_rate_base 1e-4

Requirements:
    - Distributed training with PyTorch DDP (torchrun)
    - BigEarthNet-v2 dataset in LMDB format
    - DINOv3 pretrained weights (ViT-L/16)
"""

import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import torch.nn.functional as F

from typing import List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from configilm.extra.DataSets import BENv2_DataSet
import torchvision.transforms as T

import wandb
import uuid

import torchmetrics

from utils.transforms import make_transform

from sklearn.decomposition import PCA
import io
import random
from datetime import timedelta
from contextlib import nullcontext
import argparse

from dataloaders.sen12ms_dataloader import SEN12MSDataset

def normalize_for_display(image_tensor):
    """(C, H, W) 텐서를 (H, W, C) numpy 배열로 변환하고 0-1로 정규화합니다."""
    # (C, H, W) -> (H, W, C)
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # 각 채널을 개별적으로 0-1 범위로 정규화
    img_normalized = np.zeros_like(img_np, dtype=float)
    for i in range(img_np.shape[2]):
        band = img_np[..., i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            img_normalized[..., i] = (band - band_min) / (band_max - band_min)
        else:
            img_normalized[..., i] = band
    return np.clip(img_normalized, 0, 1) # 클리핑으로 안정성 확보


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(1004)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(1004)

def visualize_batch_relation(
    student_tensor, teacher_tensor,  # [변경] 변수명 일반화 (sar_tensor -> student_tensor)
    s_features_list, t_features_list, 
    epoch, step, 
    use_teacher=True, 
    max_samples=4, 
    data_type="sar", # [추가] 데이터 타입 인자
    save_dir="./vis_results"
):
    os.makedirs(save_dir, exist_ok=True)
    
    B = min(student_tensor.shape[0], max_samples)
    
    s_data = s_features_list[-1]
    s_cls, s_patch = s_data["cls"][:B], s_data["patch"][:B]
    
    if use_teacher:
        t_data = t_features_list[-1]
        t_cls, t_patch = t_data["cls"][:B], t_data["patch"][:B]

    cols = 4 if use_teacher else 2
    fig, axs = plt.subplots(B, cols, figsize=(5 * cols, 4 * B))
    fig.suptitle(f"Relation Distillation Batch (Epoch {epoch} Step {step})", fontsize=16)
    
    if B == 1: axs = axs[np.newaxis, :]

    # ImageNet Mean/Std (Optical 복원용)
    mean = torch.tensor((0.430, 0.411, 0.296)).view(3, 1, 1).to(student_tensor.device)
    std = torch.tensor((0.213, 0.156, 0.143)).view(3, 1, 1).to(student_tensor.device)

    for idx in range(B):
        # 1. Attention Map 계산 (기존 동일)
        curr_s_cls = s_cls[idx:idx+1]
        curr_s_patch = s_patch[idx:idx+1]
        flat_s_patch = curr_s_patch.flatten(2).transpose(1, 2)
        
        attn_map_s = torch.bmm(
            F.normalize(curr_s_cls.unsqueeze(1), dim=2), 
            F.normalize(flat_s_patch, dim=2).transpose(1, 2)
        ).view(1, curr_s_patch.shape[2], curr_s_patch.shape[3])
        
        if use_teacher:
            curr_t_cls = t_cls[idx:idx+1]
            curr_t_patch = t_patch[idx:idx+1]
            flat_t_patch = curr_t_patch.flatten(2).transpose(1, 2)
            attn_map_t = torch.bmm(
                F.normalize(curr_t_cls.unsqueeze(1), dim=2), 
                F.normalize(flat_t_patch, dim=2).transpose(1, 2)
            ).view(1, curr_t_patch.shape[2], curr_t_patch.shape[3])

        img_h, img_w = student_tensor.shape[-2:]
        def min_max_norm(arr):
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            
        s_map_resized = F.interpolate(attn_map_s.unsqueeze(0), size=(img_h, img_w), mode='bicubic').squeeze().detach().cpu().numpy()
        s_map_vis = min_max_norm(s_map_resized)
        
        if use_teacher:
            t_map_resized = F.interpolate(attn_map_t.unsqueeze(0), size=(img_h, img_w), mode='bicubic').squeeze().detach().cpu().numpy()
            t_map_vis = min_max_norm(t_map_resized)

        if data_type == "sar":
            s_img = student_tensor[idx].cpu().float() * std.cpu() + mean.cpu()
            if s_img.shape[0] >= 2: s_disp = s_img[0:1]
            else: s_disp = s_img
            cmap_student = 'gray'
        else:
            opt_raw = student_tensor[idx].cpu() * std.cpu() + mean.cpu()
            s_disp = torch.clamp(opt_raw, 0, 1)
            cmap_student = None # RGB는 cmap 불필요

        student_disp = normalize_for_display(s_disp)

        teacher_disp = None
        if use_teacher:
            t_raw = teacher_tensor[idx].cpu() * std.cpu() + mean.cpu()
            teacher_disp = torch.clamp(t_raw, 0, 1)
            teacher_disp = normalize_for_display(teacher_disp)


        # 4. Plotting
        # (1) Student Input
        axs[idx, 0].imshow(student_disp, cmap=cmap_student)
        if idx == 0: axs[idx, 0].set_title(f"Student Input ({data_type.upper()})")
        axs[idx, 0].axis('off')

        # (2) Student Attention
        axs[idx, 1].imshow(student_disp, cmap=cmap_student)
        axs[idx, 1].imshow(s_map_vis, cmap='jet', alpha=0.5) 
        if idx == 0: axs[idx, 1].set_title("Student Attention")
        axs[idx, 1].axis('off')

        if use_teacher:
            # (3) Teacher Input
            axs[idx, 2].imshow(teacher_disp)
            if idx == 0: axs[idx, 2].set_title("Teacher Input (Opt)")
            axs[idx, 2].axis('off')

            # (4) Teacher Attention
            axs[idx, 3].imshow(teacher_disp)
            axs[idx, 3].imshow(t_map_vis, cmap='jet', alpha=0.5)
            if idx == 0: axs[idx, 3].set_title("Teacher Attention")
            axs[idx, 3].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close()
    return result_image

def visualize_batch_pca(
    student_tensor, teacher_tensor,
    s_patch, t_patch, 
    epoch, step, 
    use_teacher=True, 
    max_samples=4,
    data_type="sar",
    save_dir="./vis_results"
):
    os.makedirs(save_dir, exist_ok=True)
    B = min(student_tensor.shape[0], max_samples)
    
    cols = 4 if use_teacher else 2
    fig, axs = plt.subplots(B, cols, figsize=(5 * cols, 4 * B))
    fig.suptitle(f"Patch PCA Batch (Epoch {epoch} Step {step})", fontsize=16)
    if B == 1: axs = axs[np.newaxis, :]

    mean = torch.tensor((0.430, 0.411, 0.296)).view(3, 1, 1).to(student_tensor.device)
    std = torch.tensor((0.213, 0.156, 0.143)).view(3, 1, 1).to(student_tensor.device)

    for idx in range(B):
        s_feat = s_patch[idx].detach().cpu()
        C, H, W = s_feat.shape
        N = H * W
        s_flat = s_feat.view(C, -1).permute(1, 0).numpy()
        s_mean = s_flat.mean(axis=0, keepdims=True)
        s_centered = s_flat - s_mean
        
        if use_teacher:
            t_feat = t_patch[idx].detach().cpu()
            t_flat = t_feat.view(C, -1).permute(1, 0).numpy()
            t_mean = t_flat.mean(axis=0, keepdims=True)
            t_centered = t_flat - t_mean
            combined = np.concatenate([s_centered, t_centered], axis=0)
        else:
            combined = s_centered

        pca = PCA(n_components=3)
        pca.fit(combined)
        pca_feats = pca.transform(combined)
        
        pca_min = pca_feats.min(axis=0)
        pca_max = pca_feats.max(axis=0)
        denom = pca_max - pca_min
        denom[denom == 0] = 1.0
        pca_feats = (pca_feats - pca_min) / denom
        
        s_pca = pca_feats[:N]
        s_pca_img = torch.from_numpy(s_pca).view(H, W, 3).permute(2, 0, 1).unsqueeze(0).float()
        target_h, target_w = student_tensor.shape[-2:]
        s_pca_big = F.interpolate(s_pca_img, size=(target_h, target_w), mode='nearest').squeeze(0)
        s_pca_vis = s_pca_big.permute(1, 2, 0).numpy()
        
        t_pca_vis = None
        if use_teacher:
            t_pca = pca_feats[N:]
            t_pca_img = torch.from_numpy(t_pca).view(H, W, 3).permute(2, 0, 1).unsqueeze(0).float()
            t_pca_big = F.interpolate(t_pca_img, size=(target_h, target_w), mode='nearest').squeeze(0)
            t_pca_vis = t_pca_big.permute(1, 2, 0).numpy()
            
        if data_type == "sar":
            s_img = student_tensor[idx].cpu().float() * std.cpu() + mean.cpu()
            if s_img.shape[0] >= 2: s_disp = s_img[0:1]
            else: s_disp = s_img
            cmap_student = 'gray'
        else:
            opt_raw = student_tensor[idx].cpu() * std.cpu() + mean.cpu()
            s_disp = torch.clamp(opt_raw, 0, 1)
            cmap_student = None

        student_disp = normalize_for_display(s_disp)
        
        teacher_disp = None
        if use_teacher:
            t_raw = teacher_tensor[idx].cpu() * std.cpu() + mean.cpu()
            teacher_disp = torch.clamp(t_raw, 0, 1)
            teacher_disp = normalize_for_display(teacher_disp)

        # 3. Plotting
        axs[idx, 0].imshow(student_disp, cmap=cmap_student)
        if idx == 0: axs[idx, 0].set_title(f"Student Input ({data_type.upper()})")
        axs[idx, 0].axis('off')
        
        axs[idx, 1].imshow(s_pca_vis)
        if idx == 0: axs[idx, 1].set_title("Student PCA")
        axs[idx, 1].axis('off')
        
        if use_teacher:
            axs[idx, 2].imshow(teacher_disp)
            if idx == 0: axs[idx, 2].set_title("Teacher Input")
            axs[idx, 2].axis('off')
            
            axs[idx, 3].imshow(t_pca_vis)
            if idx == 0: axs[idx, 3].set_title("Teacher PCA")
            axs[idx, 3].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close()
    return result_image


def distillation_vicreg_loss(
    x: torch.Tensor, 
    y: torch.Tensor, 
    lambda_inv: float = 25.0, 
    mu_var: float = 25.0, 
    nu_cov: float = 1.0,
    eps: float = 1e-4
):

    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, C) 
    y = y.permute(0, 2, 3, 1).reshape(-1, C)

    loss_inv = F.mse_loss(x, y)

    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()

        std_x = torch.sqrt(x.var(dim=0) + eps)
        with torch.no_grad():
            target_std_y = torch.sqrt(y.var(dim=0) + eps).detach()

        loss_var = torch.mean(F.relu(target_std_y - std_x))

        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)
        
        N = x.shape[0] # N = Batch * 8 * 8 (샘플이 엄청 많아져서 통계가 더 정확해짐!)

        # 행렬 크기: (1280, N) @ (N, 1280) -> (1280, 1280)
        # 1280x1280은 GPU 입장에서 아주 작은 행렬입니다.
        cov_x = (x_centered.T @ x_centered) / (N - 1)
        with torch.no_grad():
            cov_y = ((y_centered.T @ y_centered) / (N - 1)).detach()

        loss_cov = F.mse_loss(cov_x, cov_y)

    total_loss = (lambda_inv * loss_inv +
                  mu_var * loss_var +
                  nu_cov * loss_cov)

    return total_loss, loss_inv, loss_var, loss_cov

def mld_loss_simple(student_logits, teacher_logits, T=4.0):
    with torch.no_grad():
        teacher_probs = torch.sigmoid(teacher_logits / T)

    student_logits_scaled = student_logits / T

    loss = F.binary_cross_entropy_with_logits(
        student_logits_scaled, 
        teacher_probs, 
        reduction='mean'
    )

    return loss

def gram_loss_from_maps(x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """
    x_s: Student patch map (B, C, Hs, Ws)
    x_t: Teacher patch map (B, C, Ht, Wt)
    1) 채널 방향 L2 정규화
    2) Teacher를 Student 공간 크기(Hs, Ws)로 보간
    3) (B, P, C)로 펼친 뒤 Gram = XX^T 계산
    4) Frobenius MSE
    """
    B, C, Hs, Ws = x_s.shape
    # 1) 정규화
    x_s = F.normalize(x_s, dim=1)
    # 2) 리사이즈(Teacher → Student 크기)
    if x_t.shape[-2:] != (Hs, Ws):
        x_t = F.interpolate(x_t, size=(Hs, Ws), mode='bicubic', align_corners=False)
    x_t = F.normalize(x_t, dim=1)

    # 3) (B, P, C)로 펼치기
    Xs = x_s.flatten(2).transpose(1, 2).contiguous()  # (B, P, C)
    Xt = x_t.flatten(2).transpose(1, 2).contiguous()  # (B, P, C)

    # 4) Gram = XX^T
    Gs = torch.bmm(Xs, Xs.transpose(1, 2))  # (B, P, P)
    Gt = torch.bmm(Xt, Xt.transpose(1, 2))  # (B, P, P)

    return F.mse_loss(Gs, Gt)

def _gaussian_kernel(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = torch.cdist(x, y, p=2).pow(2)
    s = torch.matmul(beta, dist.view(1, -1))
    return torch.sum(torch.exp(-s), 0)

class MK_MMDLoss(nn.Module):
    """
    Multikernel Maximum Mean Discrepancy (MK-MMD) Loss.
    """
    def __init__(self, sigmas: List[float] = [0.01, 0.1, 1., 10., 100.]):
        super(MK_MMDLoss, self).__init__()
        # sigmas를 buffer로 등록하여 GPU 이동 등을 자동으로 처리
        self.register_buffer("sigmas", torch.tensor(sigmas))

    def forward(self, x, y):
        """
        x: 첫 번째 분포의 피처맵 (B, C, H, W)
        y: 두 번째 분포의 피처맵 (B, C, H, W)
        """
        # 피처맵을 계산에 용이한 형태로 변환: (B, C, H, W) -> (B, H*W, C) -> (B*H*W, C)
        # 각 픽셀의 C차원 벡터를 하나의 샘플로 간주
        x = x.flatten(2).transpose(1, 2).reshape(-1, x.size(1))
        y = y.flatten(2).transpose(1, 2).reshape(-1, y.size(1))
        
        # MMD 계산
        xx = _gaussian_kernel(x, x, self.sigmas).mean()
        yy = _gaussian_kernel(y, y, self.sigmas).mean()
        xy = _gaussian_kernel(x, y, self.sigmas).mean()
        
        return xx + yy - 2 * xy
    
class ResidualAdapter(nn.Module):
    """
    잔차 연결을 사용하여 초기에 Identity Mapping을 수행하는 어댑터.
    F(x) 부분이 'zero-convolution'의 역할을 하여, 초기 출력은 x + 0 ≈ x 가 된다.
    """
    def __init__(self, dim: int, hidden_dim_ratio: float = 0.25):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_ratio)

        self.transform_block = nn.Sequential(
            # 채널을 줄여 파라미터 효율성을 높임
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # 다시 원래 채널로 복원
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

        # *** 핵심적인 초기화 부분 ***
        # 마지막 Conv 레이어의 가중치와 편향을 모두 0으로 초기화합니다.
        # 이렇게 하면 학습 초기에는 transform_block의 출력이 0에 가까워집니다.
        nn.init.constant_(self.transform_block[-1].weight, 0)
        nn.init.constant_(self.transform_block[-1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 x에 변환된 값을 더해줌 (Residual Connection)
        return x + self.transform_block(x)
    
class StackedMlpAdapter(nn.Module):
    """
    MlpAdapter를 여러 개 쌓아서 더 깊고 복잡한 변환을 수행하는 모듈입니다.
    """
    def __init__(self, dim, num_layers=2):
        """
        Args:
            dim (int): 입력 피처의 채널 차원
            kernel_size (int): 각 MlpAdapter의 커널 크기
            num_layers (int): 몇 개의 MlpAdapter를 쌓을 것인지 결정
        """
        super().__init__()
        
        # 지정된 num_layers 만큼 MlpAdapter를 생성하여 리스트에 담습니다.
        self.adapters = nn.ModuleList(
            [ResidualAdapter(dim, 2) for _ in range(num_layers)]
        )

    def forward(self, x):
        """
        입력 x를 각 어댑터 레이어에 순차적으로 통과시킵니다.
        
        Args:
            x (torch.Tensor): (N, C, H, W) 형태의 입력 피처
            
        Returns:
            torch.Tensor: (N, C, H, W) 형태의 최종 출력 피처
        """
        for adapter in self.adapters:
            x = adapter(x)
        return x

class ViTKDDistillationModel(nn.Module):
    """
    Teacher를 위한 원래 모델. Adapter를 포함하지 않으며,
    1D 패치 토큰을 반환합니다.
    """
    def __init__(self, backbone, num_classes: int, layers: List[int]):
        super().__init__()
        self.backbone = backbone
        d = self.backbone.embed_dim

        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
        # self.head = nn.Linear(d, num_classes)  # CLS || mean(patch)

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



        # final_idx = self.output_map[]
        _, cls_tokens_final = intermediate_outputs[-1]
        logits = self.head(cls_tokens_final)
        return logits, intermediate_features


def parse_args():
    """Parse command line arguments for 2-stage training."""
    parser = argparse.ArgumentParser(
        description="DINOv3 Knowledge Distillation Training for SAR-to-Optical Translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[0, 1],
        help="Training stage: 0 (optical baseline) or 1 (SAR with distillation)"
    )
    required.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["opt", "sar"],
        help="Data type to train on: 'opt' (optical) or 'sar' (SAR)"
    )
    required.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs"
    )

    # Path arguments
    paths = parser.add_argument_group("path arguments")
    paths.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to teacher checkpoint (required for stage 1, must be None for stage 0)"
    )
    paths.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    paths.add_argument(
        "--dinov3_repo",
        type=str,
        default="/home/hyunseo/workspace/dinov3",
        help="Path to local DINOv3 repository"
    )
    paths.add_argument(
        "--dinov3_pretrained_weights",
        type=str,
        default="/home/hyunseo/workspace/sar2opt/SAR2OPT/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        help="Path to pretrained DINOv3 weights"
    )
    paths.add_argument(
        "--dataset_images_lmdb",
        type=str,
        default="/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
        help="Path to LMDB-encoded BigEarthNet images"
    )
    paths.add_argument(
        "--dataset_metadata_parquet",
        type=str,
        default="/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
        help="Path to BigEarthNet metadata parquet file"
    )
    paths.add_argument(
        "--dataset_metadata_snow_cloud_parquet",
        type=str,
        default="/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        help="Path to BigEarthNet snow/cloud metadata parquet file"
    )

    # Dataset selection
    dataset_group = parser.add_argument_group("dataset selection")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="benv2",
        choices=["benv2", "sen12ms"],
        help="Dataset to use: 'benv2' (BigEarthNet-v2) or 'sen12ms' (SEN12MS)"
    )
    dataset_group.add_argument(
        "--sen12ms_root_dir",
        type=str,
        default="./sen12ms",
        help="Root directory for SEN12MS dataset (only used when --dataset=sen12ms)"
    )

    # Model hyperparameters
    model = parser.add_argument_group("model hyperparameters")
    model.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        help="HuggingFace model name (for reference)"
    )
    model.add_argument(
        "--label_size",
        type=int,
        default=None,
        help="Number of classes (auto-detected: BENv2=19, SEN12MS=11 if not specified)"
    )
    model.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    model.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor"
    )
    model.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    model.add_argument(
        "--layers_to_distill",
        type=int,
        nargs="+",
        default=[11, 14, 17, 20, 23],
        help="ViT layer indices to extract for distillation"
    )

    # Training hyperparameters
    training = parser.add_argument_group("training hyperparameters")
    training.add_argument(
        "--batch_size",
        type=int,
        default=72,
        help="Batch size per GPU"
    )
    training.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Total number of training epochs"
    )
    training.add_argument(
        "--learning_rate_base",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA and classification head"
    )
    training.add_argument(
        "--learning_rate_adapter",
        type=float,
        default=1e-4,
        help="Learning rate for adapter modules"
    )
    training.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler"
    )
    training.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing"
    )
    training.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    training.add_argument(
        "--resize_size",
        type=int,
        default=256,
        help="Image resize dimension (256x256)"
    )
    training.add_argument(
        "--merge_patch",
        action="store_true",
        default=True,
        help="Merge BigEarthNet patches (4 patches -> 1 image)"
    )

    # Data preprocessing
    data = parser.add_argument_group("data preprocessing")
    data.add_argument(
        "--norm_mean",
        type=float,
        nargs=3,
        default=[0.430, 0.411, 0.296],
        help="Normalization mean (R G B)"
    )
    data.add_argument(
        "--norm_std",
        type=float,
        nargs=3,
        default=[0.213, 0.156, 0.143],
        help="Normalization std (R G B)"
    )

    # Distillation hyperparameters (stage 1 only)
    distill = parser.add_argument_group("distillation hyperparameters (stage 1 only)")
    distill.add_argument(
        "--lambda_kd",
        type=float,
        default=0.1,
        help="Weight for knowledge distillation loss"
    )
    distill.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Temperature for knowledge distillation"
    )
    distill.add_argument(
        "--lambda_vicreg_inv",
        type=float,
        default=35.0,
        help="Weight for VICReg invariance loss"
    )
    distill.add_argument(
        "--lambda_vicreg_var",
        type=float,
        default=35.0,
        help="Weight for VICReg variance loss"
    )
    distill.add_argument(
        "--lambda_vicreg_cov",
        type=float,
        default=1.0,
        help="Weight for VICReg covariance loss"
    )

    # Logging and validation
    logging = parser.add_argument_group("logging and validation")
    logging.add_argument(
        "--wandb_project",
        type=str,
        default="sar2opt-dinov3",
        help="W&B project name"
    )
    logging.add_argument(
        "--validation_interval_steps",
        type=int,
        default=3000,
        help="Validation frequency in steps"
    )
    logging.add_argument(
        "--vis_interval",
        type=int,
        default=1000,
        help="Visualization frequency in steps"
    )

    # Debug options
    debug = parser.add_argument_group("debug options")
    debug.add_argument(
        "--overfit",
        type=int,
        default=0,
        choices=[0, 1],
        help="Overfit on small subset for debugging (0: disabled, 1: enabled)"
    )

    args = parser.parse_args()
    return args


args = parse_args()

# Auto-detect label_size based on dataset if not specified
if args.label_size is None:
    args.label_size = 19 if args.dataset == "benv2" else 11

dist.init_process_group("nccl", timeout=timedelta(minutes=60))
local_rank = int(os.environ['LOCAL_RANK'])
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)

# Stage validation
if args.stage == 0:
    if args.teacher_checkpoint is not None:
        raise ValueError("Stage 0 (optical baseline) should not have --teacher_checkpoint")
    if args.data_type != "opt":
        if local_rank == 0:
            print(f"WARNING: Stage 0 typically uses data_type='opt', but got '{args.data_type}'")
elif args.stage == 1:
    if args.teacher_checkpoint is None:
        raise ValueError("Stage 1 (SAR distillation) requires --teacher_checkpoint")
    if not os.path.exists(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
    if args.data_type != "sar":
        if local_rank == 0:
            print(f"WARNING: Stage 1 typically uses data_type='sar', but got '{args.data_type}'")

use_teacher = (args.stage == 1)

# Initialize training state
start_epoch = 1
checkpoint = None
wandb_run_id = None

if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
    checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch']
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    if local_rank == 0:
        print(f"Found checkpoint. Resuming from {args.resume_from_checkpoint}")
        print(f"Resuming from epoch {start_epoch}, wandb run ID: {wandb_run_id}")
else:
    if local_rank == 0:
        if args.resume_from_checkpoint:
            print(f"Checkpoint path provided but not found: {args.resume_from_checkpoint}")
        print("Starting training from scratch.")

# Setup dataset paths
datapath = {
    "images_lmdb": args.dataset_images_lmdb,
    "metadata_parquet": args.dataset_metadata_parquet,
    "metadata_snow_cloud_parquet": args.dataset_metadata_snow_cloud_parquet,
}

# LoRA target modules (fixed for DINOv3)
LORA_TARGET_MODULES = ["attn.qkv", "attn.proj"]

if local_rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)

    if wandb_run_id:
        wandb.init(
            project=args.wandb_project,
            id=wandb_run_id,
            resume="must",
        )
    else:
        # New training run
        random_id = uuid.uuid4().hex[:8]
        wandb.init(
            project=args.wandb_project,
            name=f"stage{args.stage}_{args.data_type}_lr{args.learning_rate_base}_bs{args.batch_size}_{random_id}",
            resume="allow",
            tags=[f"stage{args.stage}", args.data_type],
            config=vars(args)
        )
    print(f"Using device: {device} (DDP local_rank: {local_rank})")
    print(f"Training configuration: Stage {args.stage}, Data type: {args.data_type}")
    print(f"Output directory: {args.output_dir}")


def validate_and_log(model, val_loader, device, num_classes, global_step, data_type, dataset):
    model.eval()

    apm_metric = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average='macro').to(device)
    apmu_metric = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average='micro').to(device)
    fm1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro').to(device)
    fmu1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='micro').to(device)

    val_progress_bar = tqdm(val_loader, desc="Validating", disable=(dist.get_rank() != 0))

    with torch.inference_mode():
        for batch in val_progress_bar:
            if dataset == "benv2":
                img, lbl, _ = batch
            else:
                img, lbl = batch

            if data_type == "sar":
                inputs = img["sar"].to(device)
            elif data_type == "opt":
                inputs = img["opt"].to(device)
            else:
                raise ValueError(f"Invalid data_type: {data_type}")
            
            labels = lbl.to(device).int() 
            
            with torch.cuda.amp.autocast():
                logits, _ = model(inputs)
            
            preds = torch.sigmoid(logits)
            
            apm_metric.update(preds, labels)
            apmu_metric.update(preds, labels)
            fm1_metric.update(preds, labels)
            fmu1_metric.update(preds, labels)

    # 3. 모든 Rank가 .update()를 마칠 때까지 대기 (선택 사항이지만 안전함)
    dist.barrier() 
    
    # 4. [중요] .compute()는 "if rank == 0" *밖에서* 모든 Rank가 호출해야 함
    # DDP 동기화가 여기서 발생합니다.
    apm = apm_metric.compute()
    apmu = apmu_metric.compute()
    fm1 = fm1_metric.compute()
    fmu1 = fmu1_metric.compute()

    # 5. [중요] .reset()도 모든 Rank가 호출해야 함
    apm_metric.reset()
    apmu_metric.reset()
    fm1_metric.reset()
    fmu1_metric.reset()
    
    # 6. 로깅 및 출력은 Rank 0에서만 수행
    if dist.get_rank() == 0:
        print("\n--- Calculating Metrics on Rank 0 ---")
        print(f"Validation Step: {global_step}")
        print(f"APM: {apm:.4f}, APμ: {apmu:.4f}, FM1: {fm1:.4f}, Fμ1: {fmu1:.4f}")

        wandb.log({
            "validation/APM": apm,
            "validation/APμ": apmu,
            "validation/FM1": fm1,
            "validation/Fμ1": fmu1,
        }, step=global_step)
        
    # 7. 모든 Rank가 로깅까지 마칠 때까지 대기
    dist.barrier()
    model.train()

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=args.lora_dropout,
    bias="none",
)

if use_teacher:
    if local_rank == 0:
        print(f"Loading teacher model from {args.teacher_checkpoint}")
    teacher_backbone = torch.hub.load(args.dinov3_repo, 'dinov3_vitl16', source='local', weights=args.dinov3_pretrained_weights)
    teacher_model = ViTKDDistillationModel(teacher_backbone, num_classes=args.label_size, layers=args.layers_to_distill).to(device)
    teacher_model = get_peft_model(teacher_model, lora_config)

    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location='cpu')['model_state_dict']
    teacher_model.load_state_dict(teacher_ckpt, strict=False)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    if local_rank == 0:
        print("Teacher model loaded and frozen successfully")
else:
    teacher_model = None
    if local_rank == 0:
        print("Stage 0: Training without teacher model")


student_backbone = torch.hub.load(args.dinov3_repo, 'dinov3_vitl16', source='local', weights=args.dinov3_pretrained_weights)
model = ViTKDDistillationModel(student_backbone, num_classes=args.label_size, layers=args.layers_to_distill).to(device)
model = get_peft_model(model, lora_config)
model = DDP(model, device_ids=[local_rank])

adapter_dim = student_backbone.embed_dim
mimicking_adapters = nn.ModuleList([
    DDP(
        StackedMlpAdapter(adapter_dim, num_layers=1).to(device),
        device_ids=[local_rank]
    )
    for _ in args.layers_to_distill
])

for name, param in model.module.named_parameters():
    param.requires_grad = False
    if 'base_model.model.head.' in name:
        param.requires_grad = True
        continue
    if 'lora' in name:
        param.requires_grad = True
        continue

lora_params = []
head_params = []
for name, param in model.module.named_parameters():
    param.requires_grad = False
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    
    if 'base_model.model.head.' in name:
        param.requires_grad = True
        head_params.append(param)
    

adapter_params = []
for adapter in mimicking_adapters:
    adapter_params.extend(list(adapter.parameters()))


opt_params = [
    {
        "params": lora_params,
        "lr": args.learning_rate_base
    },
    {
        "params": head_params,
        "lr": args.learning_rate_base
    },
    {
        "params": adapter_params,
        "lr": args.learning_rate_adapter
    }
]
optimizer_student = torch.optim.Adam(opt_params, lr=args.learning_rate_base)


transform = {
    "opt": make_transform(resize_size=args.resize_size, data_type="opt", is_train=True, train_datatype="opt", dataset=args.dataset),
    "sar": make_transform(
        resize_size=args.resize_size,
        data_type="sar",
        is_train=True,
        train_datatype="sar",
        dataset=args.dataset
    )
}

transform_val = {
    "opt": make_transform(resize_size=args.resize_size, data_type="opt", is_train=False, train_datatype="opt", dataset=args.dataset),
    "sar": make_transform(
        resize_size=args.resize_size,
        data_type="sar",
        is_train=False,
        train_datatype="sar",
        dataset=args.dataset
    )
}


if args.dataset == "benv2":
    train_dataset = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='train',
        transform=transform,
        merge_patch=args.merge_patch
    )
    val_dataset = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='test',
        transform=transform_val,
        merge_patch=args.merge_patch,
        get_labels_name=True
    )
elif args.dataset == "sen12ms":
    train_dataset = SEN12MSDataset(
        root_dir=args.sen12ms_root_dir,
        subset="train",
        seed=42,
        transform=transform
    )
    val_dataset = SEN12MSDataset(
        root_dir=args.sen12ms_root_dir,
        subset="test",
        seed=42,
        transform=transform_val
    )

pos_weight = train_dataset.get_class_weights()
pos_weight = pos_weight.to(device)
print("Class Weights:", pos_weight)

if args.overfit == 1:
    base_indices = list(range(min(args.batch_size, len(train_dataset))))
    replicated_indices = base_indices * 10000
    train_dataset = Subset(train_dataset, replicated_indices)
    if local_rank == 0:
        print(f"WARNING: Overfitting mode enabled with {len(base_indices)} samples")

train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False) # 검증 시에는 셔플 불필요

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=train_sampler,
    worker_init_fn=seed_worker,
    generator=g
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=val_sampler
)

scheduler_student = CosineAnnealingLR(optimizer_student, T_max=args.num_epochs, eta_min=args.eta_min)
scaler = torch.cuda.amp.GradScaler()

dist.barrier()

bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
    if local_rank == 0:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}...")

    checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
    
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer_student.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_student_state_dict' in checkpoint:
        scheduler_student.load_state_dict(checkpoint['scheduler_student_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    start_epoch = checkpoint['epoch']
    if 'wandb_run_id' in checkpoint:
        wandb_run_id = checkpoint['wandb_run_id']
        
    if local_rank == 0:
        print(f"Successfully resumed from Epoch {checkpoint['epoch']}")

    del checkpoint
    torch.cuda.empty_cache()

global_step = (start_epoch - 1) * len(train_dataloader)

if local_rank == 0:
    print("Fetching fixed validation batch for visualization...")
    val_iter = iter(val_dataloader)
    if args.dataset == "benv2":
        fixed_val_img, fixed_val_lbl, fixed_val_lbl_names = next(val_iter)
    else:
        fixed_val_img, fixed_val_lbl = next(val_iter)

    fixed_sar_val = fixed_val_img["sar"].to(device)
    fixed_opt_val = fixed_val_img["opt"].to(device) 



for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):

    model.train()

    train_sampler.set_epoch(epoch)


    total_loss_epoch, total_task_loss_epoch, total_adv_loss_epoch = 0, 0, 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{start_epoch + args.num_epochs}", disable=(local_rank != 0))
    
    optimizer_student.zero_grad()
    
    for i, (img, lbl) in enumerate(progress_bar):
        is_update_step = ((i + 1) % args.gradient_accumulation_steps == 0) or ((i + 1) == len(train_dataloader))
        my_context = model.no_sync if not is_update_step else nullcontext

        if global_step < args.warmup_steps:
            lr_scale = global_step / args.warmup_steps
            for i, param_group in enumerate(optimizer_student.param_groups):
                base_lr = scheduler_student.base_lrs[i]
                param_group['lr'] = base_lr * lr_scale

        sar_inputs = img["sar"].to(device)
        opt_inputs = img["opt"].to(device)
        labels = lbl.to(device).float()

        with torch.cuda.amp.autocast():
            student_logits, student_raw_features = model(sar_inputs if args.data_type == "sar" else opt_inputs)

            loss_task = bce_criterion(student_logits, labels)

            if use_teacher:
                with torch.no_grad():
                    teacher_logits, teacher_raw_features = teacher_model(opt_inputs)

                loss_classifier_kd = mld_loss_simple(
                    student_logits,
                    teacher_logits,
                    T=args.temperature
                )

                loss_md_total = 0.0
                loss_reg_total = 0.0
                loss_attn_total = 0.0
                loss_cls_total = 0.0

                layer_stats = {}
                
                total_s_std = 0.0
                total_t_std = 0.0
                total_cos_sim = 0.0

                for i, layer_idx in enumerate(args.layers_to_distill):
                    s_data = student_raw_features[layer_idx]
                    t_data = teacher_raw_features[layer_idx]

                    s_patch = s_data["patch"]
                    t_patch = t_data["patch"]
                    s_cls = s_data["cls"].unsqueeze(1)
                    t_cls = t_data["cls"].unsqueeze(1)

                    layer_cls_loss = F.mse_loss(s_cls, t_cls)
                    loss_cls_total += layer_cls_loss

                    l_vicreg, l_inv, l_var, l_cov = distillation_vicreg_loss(
                        s_patch,
                        t_patch,
                        lambda_inv=args.lambda_vicreg_inv,
                        mu_var=args.lambda_vicreg_var,
                        nu_cov=args.lambda_vicreg_cov
                    )
                    loss_reg_total += l_vicreg

                    B, C, H, W = s_patch.shape
                    N = H * W
                    s_patch_tokens = s_patch.flatten(2).transpose(1, 2)
                    t_patch_tokens = t_patch.flatten(2).transpose(1, 2)

                    s_cls_norm = F.normalize(s_cls, dim=2)
                    s_patch_tokens_norm = F.normalize(s_patch_tokens, dim=2) 

                    t_cls_norm = F.normalize(t_cls, dim=2)
                    t_patch_tokens_norm = F.normalize(t_patch_tokens, dim=2)

                    s_attn_map = torch.bmm(s_cls_norm, s_patch_tokens_norm.transpose(1, 2))
                    t_attn_map = torch.bmm(t_cls_norm, t_patch_tokens_norm.transpose(1, 2))
                    
                    scale_factor = 20.0 

                    s_logits = s_attn_map * scale_factor
                    t_logits = t_attn_map * scale_factor

                    s_log_probs = F.log_softmax(s_logits, dim=-1)
                    t_probs = F.softmax(t_logits, dim=-1)
                    
                    layer_attn_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean')
                    loss_attn_total += layer_attn_loss
                    with torch.no_grad():
                        B, C, H, W = s_patch.shape

                        # (N, C) 형태로 변환
                        s_reshaped = s_patch.permute(0, 2, 3, 1).reshape(-1, C)
                        t_reshaped = t_patch.permute(0, 2, 3, 1).reshape(-1, C)

                        # 현재 Layer의 통계 계산
                        curr_s_std = s_reshaped.std(dim=0).mean().item()
                        curr_t_std = t_reshaped.std(dim=0).mean().item()
                        curr_cos_sim = F.cosine_similarity(s_reshaped, t_reshaped, dim=1).mean().item()
                        
                        # 전체 평균용 누적
                        total_s_std += curr_s_std
                        total_t_std += curr_t_std
                        total_cos_sim += curr_cos_sim

                        # Layer별 통계 저장 (키 이름: layer_XX_...)
                        # layer_idx를 두 자리 수로 포맷팅 (01, 02...)하여 정렬 예쁘게
                        layer_key = f"layer_{layer_idx:02d}"
                        layer_stats[f"debug/{layer_key}_s_std"] = curr_s_std
                        layer_stats[f"debug/{layer_key}_t_std"] = curr_t_std
                        layer_stats[f"debug/{layer_key}_cos_sim"] = curr_cos_sim
                        layer_stats[f"debug/{layer_key}_std_ratio"] = curr_s_std / (curr_t_std + 1e-6)

                    if args.layers_to_distill:
                        num_layers = len(args.layers_to_distill)

                        loss_reg_total /= num_layers
                        loss_attn_total /= num_layers

                        avg_s_std = total_s_std / num_layers
                        avg_t_std = total_t_std / num_layers
                        avg_cos_sim = total_cos_sim / num_layers

                loss_classifier = loss_task + args.lambda_kd * loss_classifier_kd
                loss_attn = 0.1 * (loss_attn_total + loss_cls_total)
                loss_reg = loss_reg_total

                total_loss = loss_reg + loss_attn + loss_classifier
            else:
                loss_classifier = loss_task
                loss_classifier_kd = torch.tensor(0.0)
                loss_cls_total = torch.tensor(0.0)

                loss_vicreg = torch.tensor(0.0)
                loss_attn_total = torch.tensor(0.0)

                loss_reg = torch.tensor(0.0)
                loss_attn = torch.tensor(0.0)
                loss_classifier = loss_task

                avg_s_std = 0.0
                avg_t_std = 0.0
                avg_cos_sim = 0.0
                
                layer_stats = {}


                total_loss = loss_classifier

            with my_context():
                total_loss = total_loss / args.gradient_accumulation_steps
                scaler.scale(total_loss).backward()
        

        if is_update_step:
            global_step += 1

            scaler.step(optimizer_student)
            scaler.update()
            optimizer_student.zero_grad()

            total_loss_log = total_loss * args.gradient_accumulation_steps
            total_loss_epoch += total_loss_log.item()
            
            if local_rank == 0:
                current_lr_base = optimizer_student.param_groups[0]['lr']
                current_lr_adapter = optimizer_student.param_groups[2]['lr']
                progress_bar.set_postfix({
                    'total_loss': f'{total_loss_log.item():.4f}',
                    'loss_classifier_kd': f'{loss_classifier_kd.item():.4f}',
                    'task_loss': f'{loss_task.item():.4f}',

                    'loss_attn_total': f'{loss_attn_total.item():.4f}',
                    'loss_cls_total': f'{loss_cls_total.item():.4f}',

                    'loss_vicreg': f'{loss_reg.item():.4f}',
                    'lr_base': f'{current_lr_base:.2e}',
                    'lr_adapter': f'{current_lr_adapter:.2e}',
                })
                log_dict = {
                    "train/total_loss": total_loss_log.item(),
                    "train/loss_classifier_kd": loss_classifier_kd.item(),
                    "train/task_loss": loss_task.item(),

                    "train/loss_attn_total": loss_attn_total.item(),
                    "train/loss_cls_total": loss_cls_total.item(),

                    "train/loss_vicreg": loss_reg.item(),

                    "train/learning_rate_base": current_lr_base,
                    "train/learning_rate_adapter": current_lr_adapter,

                    "debug/avg_student_std": avg_s_std,
                    "debug/avg_teacher_std": avg_t_std,
                    "debug/avg_cosine_sim": avg_cos_sim,
                    "debug/avg_std_ratio": avg_s_std / (avg_t_std + 1e-6),
                }

                log_dict.update(layer_stats)

                wandb.log(log_dict, step=global_step)

                if global_step % args.vis_interval == 0:
                    with torch.no_grad():
                        N_VIS = 4

                        vis_sar = fixed_sar_val[:N_VIS]
                        vis_opt = fixed_opt_val[:N_VIS]

                        _, s_vis_output = model.module(vis_sar if args.data_type == "sar" else vis_opt)

                        if use_teacher:
                            t_logit, t_vis_output = teacher_model(vis_opt)
                        else:
                            t_vis_output = None

                        s_feats_vis = []
                        t_feats_vis = []
                        for layer_idx in args.layers_to_distill[-1:]:
                            s_feats_vis.append(s_vis_output[layer_idx])
                            if use_teacher:
                                t_feats_vis.append(t_vis_output[layer_idx])
                            else:
                                t_feats_vis.append(None)

                        vis_img = visualize_batch_relation(
                            vis_sar if args.data_type == "sar" else vis_opt,
                            vis_opt,
                            s_feats_vis,
                            t_feats_vis,
                            epoch,
                            global_step,
                            use_teacher=use_teacher,
                            max_samples=N_VIS,
                            save_dir=os.path.join(args.output_dir, "attention_vis_results"),
                            data_type=args.data_type
                        )
                        wandb.log({"inference/relation_map_batch": wandb.Image(vis_img)}, step=global_step)

                        last_layer_idx = args.layers_to_distill[-1]
                        s_patch_input = s_vis_output[last_layer_idx]["patch"]
                        t_patch_input = t_vis_output[last_layer_idx]["patch"] if use_teacher else None

                        vis_pca_img = visualize_batch_pca(
                            vis_sar if args.data_type == "sar" else vis_opt,
                            vis_opt,
                            s_patch_input,
                            t_patch_input,
                            epoch,
                            global_step,
                            use_teacher=use_teacher,
                            max_samples=N_VIS,
                            data_type=args.data_type,
                            save_dir=os.path.join(args.output_dir, "pca_vis_results")
                        )
                        wandb.log({"inference/patch_pca_batch": wandb.Image(vis_pca_img)}, step=global_step)

                        ##############################################
                        ##############################################
                        ##############################################
                        ##############################################
                        ##############################################
                        ##############################################
                        ##############################################


                        s_feats_train = []
                        t_feats_train = []
                        for layer_idx in args.layers_to_distill[-1:]:
                            s_feats_train.append(student_raw_features[layer_idx])
                            if use_teacher:
                                t_feats_train.append(teacher_raw_features[layer_idx])
                            else:
                                t_feats_train.append(None)

                        # Train Vis: Relation
                        train_vis_img = visualize_batch_relation(
                            sar_inputs if args.data_type == "sar" else opt_inputs,
                            opt_inputs,
                            s_feats_train,
                            t_feats_train,
                            epoch,
                            global_step,
                            use_teacher=use_teacher,
                            max_samples=N_VIS,
                            save_dir=os.path.join(args.output_dir, "train_vis_results"),
                            data_type=args.data_type
                        )
                        wandb.log({"train/relation_map_batch": wandb.Image(train_vis_img)}, step=global_step)

                        # Train Vis: PCA
                        s_patch_train = student_raw_features[last_layer_idx]["patch"]
                        t_patch_train = teacher_raw_features[last_layer_idx]["patch"] if use_teacher else None

                        train_vis_pca = visualize_batch_pca(
                            sar_inputs if args.data_type == "sar" else opt_inputs,
                            opt_inputs,
                            s_patch_train,
                            t_patch_train,
                            epoch,
                            global_step,
                            use_teacher=use_teacher,
                            max_samples=N_VIS,
                            data_type=args.data_type,
                            save_dir=os.path.join(args.output_dir, "train_vis_results")
                        )
                        wandb.log({"train/patch_pca_batch": wandb.Image(train_vis_pca)}, step=global_step)

                        torch.cuda.empty_cache()

        del student_logits, student_raw_features, loss_task
        if use_teacher:
            del teacher_logits, teacher_raw_features, loss_reg, loss_attn, loss_classifier
            del l_vicreg, l_inv, l_var, l_cov

    del total_loss
    avg_loss = total_loss_epoch / (len(train_dataloader) // args.gradient_accumulation_steps)

    if global_step >= args.warmup_steps:
        scheduler_student.step()
    torch.cuda.empty_cache()
    validate_and_log(model, val_dataloader, device, args.label_size, global_step, args.data_type, args.dataset)
    torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_loss": avg_loss, "epoch": epoch + 1})

        if epoch % 1 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_stage{args.stage}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer_student.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_student_state_dict': scheduler_student.state_dict(),
                'global_step': global_step,
                'wandb_run_id': wandb.run.id,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    dist.barrier()  

if local_rank == 0:
    print("\n--- Training Finished ---")

dist.destroy_process_group()