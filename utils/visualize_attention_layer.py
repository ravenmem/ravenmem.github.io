import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model
from utils.transforms import make_transform
from configilm.extra.DataSets import BENv2_DataSet
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torch import Tensor, nn

import types

class DinoV3Linear(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.head = nn.Linear(hidden_size, num_classes)

    def forward_features(self, pixel_values):
        # Backbone을 통해 feature만 추출
        return self.backbone(pixel_values)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        logits = self.head(outputs)
        return logits

attention_maps = []


def new_compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
    """
    이 함수는 'SelfAttention' 인스턴스의 'compute_attention' 메서드를 대체합니다.
    'self'는 SelfAttention 인스턴스 자신을 가리킵니다.
    """
    # 원본 compute_attention의 전반부 (q, k, v 준비)
    assert attn_bias is None
    B, N, _ = qkv.shape
    C = self.qkv.in_features
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)

    # --- F.scaled_dot_product_attention 대체 ---
    # 1. (Q @ K^T) * scale 수동 계산
    # 'self.scale'은 SelfAttention.__init__에서 정의된 값입니다.
    attn = (q @ k.transpose(-2, -1)) * self.scale
    
    attn = F.softmax(attn, dim=-1)
    
    attention_maps.append(attn.detach())
    
    # 4. @ V 계산
    x = attn @ v
    # --- 대체 완료 ---

    # 원본 compute_attention의 후반부 (reshape)
    x = x.transpose(1, 2)
    return x.reshape([B, N, C])


LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv"]

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
)

REPO_NAME = "/home/hyunseo/workspace/dinov3"  
DINOV3_PRETRAINED_WEIGHTS = '/home/hyunseo/workspace/sar2opt/SAR2OPT/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
LABEL_SIZE = 19
DATA_TYPE = "sar"  

# INFERENCE_CHECKPOINT_PATH = f"./checkpoints/stage0_{DATA_TYPE}/checkpoint_stage0_epoch101.pth"
INFERENCE_CHECKPOINT_PATH = f"./checkpoints/stage0_opt/checkpoint_stage0_epoch101.pth"

device = f"cuda:0"

cls_feature_size = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local').num_features
backbone = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local', weights=DINOV3_PRETRAINED_WEIGHTS)
model = DinoV3Linear(backbone, hidden_size=cls_feature_size, num_classes=LABEL_SIZE, freeze_backbone=True).to(device)
model = get_peft_model(model, lora_config).to(device)
student_ckpt = torch.load(INFERENCE_CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(student_ckpt['model_state_dict'], strict=False)


for name, param in model.named_parameters():
    print(name)

for block in model.base_model.model.backbone.blocks:
    try:
        block.attn.compute_attention = types.MethodType(new_compute_attention, block.attn)
    except AttributeError:
        print("Warning: 'attn' attribute not found in a block.")
# for block in model.backbone.blocks:
#     try:
#         block.attn.compute_attention = types.MethodType(new_compute_attention, block.attn)
#     except AttributeError:
#         print("Warning: 'attn' attribute not found in a block.")


datapath = {
    "images_lmdb": "/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
    "metadata_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

MERGE_PATCH = False
RESIZE_SIZE = 256 if MERGE_PATCH else 128

transform_val = {
    "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False),
    "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=False)
}

val_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    img_size=(12, 120, 120),
    split='test',
    transform=transform_val,
    merge_patch=MERGE_PATCH,
    max_len=1000
)

BATCH_SIZE = 24
val_dataloader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE ,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

img_tensor = next(iter(val_dataloader))
img = img_tensor[0][DATA_TYPE].to(device)
lbl = img_tensor[1]

with torch.no_grad():
    output = model(img)

print(lbl[0])
print(torch.sigmoid(output[0]))

def visualize_attention(image, attention_maps, layer_indices, batch_idx=0):
    """
    (수정) 특정 레이어의 '패치-to-패치' 64x64 어텐션 맵을 시각화하는 함수
    """
    # --- 원본 이미지 정규화 해제 (이제 필요 없음) ---
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    # image_for_viz = unnormalize(image[batch_idx]).permute(1, 2, 0).cpu().numpy()

    # --- 토큰 인덱스 계산 ---
    
    # N (num_tokens) = 69
    num_tokens = attention_maps[0].shape[-1]
    
    # 1. 실제 패치 토큰의 개수 계산 (8x8=64)
    h_featmap = w_featmap = image.shape[-1] // 16
    num_patch_tokens = h_featmap * w_featmap
    
    # 2. 스토리지 토큰 개수 계산 (4)
    num_storage_tokens = (num_tokens - 1) - num_patch_tokens
    
    # 3. (신규) 패치 토큰이 시작되는 인덱스 계산
    # [CLS] (1개) + [STORAGE] (4개) = 5. (즉, 5번 인덱스부터 패치 토큰)
    patch_start_index = 1 + num_storage_tokens
    
    # --- 시각화 ---

    num_layers = len(layer_indices)
    fig, axes = plt.subplots(1, num_layers, figsize=(5.5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes] 

    im = None # imshow 이미지를 저장할 변수

    for i, layer_idx in enumerate(layer_indices):
        attn_map_layer = attention_maps[layer_idx] # (B, H, N, N)
        
        # 1. batch_idx로 (H, N, N) 텐서를 먼저 선택
        attn_map_batch = attn_map_layer[batch_idx]
        
        # 2. 헤드 평균 (H, N, N) -> (N, N)
        attn_map_avg_heads = attn_map_batch.mean(dim=0)
        
        # 3. (수정) (N, N) 맵에서 (64, 64) 패치-to-패치 맵만 슬라이싱
        #    토큰 5번~끝 (Query)과 토큰 5번~끝 (Key)을 선택
        patch_to_patch_attn = attn_map_avg_heads[patch_start_index:, patch_start_index:]
        
        # 4. (수정) 64x64 히트맵을 직접 그립니다.
        #    .cpu()와 .numpy()가 필요합니다.
        im = axes[i].imshow(patch_to_patch_attn.cpu().numpy(), cmap='jet')
        axes[i].set_title(f"Layer {layer_idx} Patch-to-Patch (64x64)")
        axes[i].axis('off')

    # (신규) 그림 오른쪽에 컬러바(색상 막대) 추가
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    plt.savefig(f"./save_fig/opt_sar/attention_visualization_64x64_batch{batch_idx}.png")
print('attention_maps: ', len(attention_maps), attention_maps[0].shape)


if attention_maps:
    for b_idx in range(img.size(0)):
        visualize_attention(img, attention_maps, layer_indices=list(range(0, 32)), batch_idx=b_idx)