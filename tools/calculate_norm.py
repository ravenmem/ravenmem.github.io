import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np


from configilm.extra.DataSets import BENv2_DataSet
from utils.transforms import make_transform

datapath = {
    "images_lmdb": "/root/hyun/rico-hdl/Encoded-BigEarthNet",
    "metadata_parquet": "/root/hyun/meta/metadata.parquet",
    "metadata_snow_cloud_parquet": "/root/hyun/meta/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BATCH_SIZE = 512
NUM_WORKERS = 4
# -----------------------------------------------------------

def refined_lee_filter_pytorch(img_tensor, kernel_size=3, k=6):
    """
    Refined Lee Filter (RLF) - GPU Optimized
    """
    if k > kernel_size * kernel_size:
        raise ValueError(f"k ({k}) cannot be larger than the window size ({kernel_size*kernel_size})")

    H, W = img_tensor.shape[-2:]
    pad = kernel_size // 2

    if img_tensor.dim() == 2:
        img_in = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.dim() == 3:
        img_in = img_tensor.unsqueeze(0)
    else:
        img_in = img_tensor

    # 1. Unfold
    patches = F.unfold(img_in, kernel_size=kernel_size, padding=pad)
    patches = patches.view(img_in.size(0), img_in.size(1), kernel_size*kernel_size, -1)
    
    # 2. Center Pixel
    center_idx = (kernel_size * kernel_size) // 2
    center_pixels = patches[:, :, center_idx, :].unsqueeze(2)

    # 3. Distance & Top-K
    distances = torch.abs(patches - center_pixels)
    _, topk_indices = torch.topk(distances, k, dim=2, largest=False, sorted=False)

    # 4. Gather
    knn_patches = torch.gather(patches, 2, topk_indices)

    # 5. Statistics
    rlf_local_mean = torch.mean(knn_patches, dim=2)
    rlf_local_var = torch.var(knn_patches, dim=2)

    # 6. Filtering
    overall_var = torch.mean(rlf_local_var) 
    epsilon = 1e-8
    k_weight = rlf_local_var / (rlf_local_var + overall_var + epsilon)
    
    filtered_values = rlf_local_mean + k_weight * (center_pixels.squeeze(2) - rlf_local_mean)
    
    result = filtered_values.view(img_in.size(0), img_in.size(1), H, W)

    if img_tensor.dim() == 2:
        return result.squeeze(0).squeeze(0)
    elif img_tensor.dim() == 3:
        return result.squeeze(0)
    return result

def calculate_benv2_sar_stats_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_size = 256

    transform = {
        "sar": {
            "preprocess": v2.Compose([
                v2.ToTensor(),
                v2.Resize((resize_size, resize_size), antialias=True),
            ]),
            "augment": v2.Identity(),
            "normalize": v2.Compose([v2.ToDtype(torch.float32, scale=False)])
        },
        "opt": {
            "preprocess": v2.Compose([
                v2.ToTensor(),
                v2.Resize((resize_size, resize_size), antialias=True),
            ]),
            "augment": v2.Identity(),
            "normalize": v2.Compose([v2.ToDtype(torch.float32, scale=False)])
        }
    }

    print("훈련 데이터셋을 로딩 중입니다...")
    train_dataset = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='train',
        transform=transform,
        merge_patch=True,
    )

    calculate_global_percentiles(train_dataset, num_samples=300000)

def get_quantile_from_hist(hist, bins, min_val, max_val, q):
    """
    누적된 히스토그램에서 분위수(Quantile) 값을 역산합니다.
    """
    # 1. 누적 분포 함수(CDF) 생성 (Counts 누적)
    cdf = torch.cumsum(hist, dim=0)
    total_count = cdf[-1]
    
    # 2. 목표 랭크(순위) 계산
    target_rank = total_count * q
    
    # 3. 목표 랭크를 넘어서는 첫 번째 bin 인덱스 찾기
    # searchsorted가 빠르지만, cdf가 정렬되어 있으므로 단순 비교
    idx = torch.searchsorted(cdf, target_rank)
    
    # 4. Bin 인덱스를 실제 값으로 변환
    # bin_width = (max - min) / bins
    bin_width = (max_val - min_val) / bins
    value = min_val + (idx * bin_width)
    return value.item()

def calculate_global_percentiles(dataset, num_samples=5000):
    num_samples = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----------------------------------------------------------------
    # 1. 히스토그램 설정 (정확도를 위해 bin을 촘촘하게 설정)
    # ----------------------------------------------------------------
    # SAR 예상 범위: -100dB ~ +50dB (매우 넉넉하게 잡음)
    # Optical 예상 범위: 0.0 ~ 1.0 (Normalization 가정) 또는 0 ~ 10000 (Raw)
    
    SAR_MIN, SAR_MAX = -100.0, 50.0
    OPT_MIN, OPT_MAX = 0.0, 10000.0 # Tensor 변환 시 0~1 범위라고 가정
    
    NUM_BINS = 30000 # Bin이 많을수록 정확해집니다 (3만개면 0.005 단위 정밀도)
    
    # 채널별 히스토그램 초기화 (B, G, R 등 채널 수 3개 가정)
    hist_sar = torch.zeros(3, NUM_BINS).to(device)
    hist_opt = torch.zeros(3, NUM_BINS).to(device)
    
    print(f"히스토그램 방식을 사용하여 {num_samples}개 샘플의 정확한 분포를 계산합니다...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, _ = batch
            
            sar_batch = imgs['sar'].to(device)
            opt_batch = imgs['opt'].to(device)
            
            # --- SAR 전처리 ---
            sar_linear = 10 ** (sar_batch / 10.0)
            sar_filtered_linear = refined_lee_filter_pytorch(sar_linear) 
            sar_filtered_db = 10.0 * torch.log10(sar_filtered_linear + 1e-6)
            
            vv = sar_filtered_db[:, 0:1, :, :]
            vh = sar_filtered_db[:, 1:2, :, :]
            diff = vv - vh
            sar_final = torch.cat([vv, vh, diff], dim=1)
            
            # --- 히스토그램 누적 (핵심) ---
            for c in range(3):
                # SAR 채널 c
                hist_sar[c] += torch.histc(
                    sar_final[:, c, :, :].float(), 
                    bins=NUM_BINS, 
                    min=SAR_MIN, 
                    max=SAR_MAX
                )
                
                # Optical 채널 c
                hist_opt[c] += torch.histc(
                    opt_batch[:, c, :, :].float(), 
                    bins=NUM_BINS, 
                    min=OPT_MIN, 
                    max=OPT_MAX
                )

    print("수집 완료. Quantile 역산 중...")
    
    sar_min_list = []
    sar_max_list = []
    opt_min_list = []
    opt_max_list = []
    
    for c in range(3):
        # SAR Quantile (2%, 98%)
        s_min = get_quantile_from_hist(hist_sar[c], NUM_BINS, SAR_MIN, SAR_MAX, 0.001)
        s_max = get_quantile_from_hist(hist_sar[c], NUM_BINS, SAR_MIN, SAR_MAX, 0.999)
        sar_min_list.append(s_min)
        sar_max_list.append(s_max)
        
        # Opt Quantile (2%, 98%)
        o_min = get_quantile_from_hist(hist_opt[c], NUM_BINS, OPT_MIN, OPT_MAX, 0.001)
        o_max = get_quantile_from_hist(hist_opt[c], NUM_BINS, OPT_MIN, OPT_MAX, 0.999)
        opt_min_list.append(o_min)
        opt_max_list.append(o_max)

    result = {
        "sar_min": sar_min_list,
        "sar_max": sar_max_list,
        "opt_min": opt_min_list,
        "opt_max": opt_max_list
    }
    
    print("계산 결과:", result)
    return result
if __name__ == "__main__":
    torch.cuda.empty_cache()
    calculate_benv2_sar_stats_gpu()