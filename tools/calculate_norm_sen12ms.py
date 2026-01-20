import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np
import rasterio
import glob
import os

# -----------------------------------------------------------
# [설정] SEN12MS 데이터셋 경로
# -----------------------------------------------------------
SEN12MS_ROOT = "./sen12ms"
BATCH_SIZE = 256  # GPU 메모리에 맞춰 조절
NUM_WORKERS = 8
NUM_SAMPLES = 40000 # 전체 데이터 중 통계를 낼 샘플 수 (많을수록 정확함)

# -----------------------------------------------------------
# 1. Raw Data 로딩용 데이터셋 클래스 (정규화 X)
# -----------------------------------------------------------
class RawSEN12MSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # S1 파일 경로 리스트 생성
        self.s1_paths = glob.glob(os.path.join(root_dir, "*", "s1", "*.tif"))
        self.data_pairs = []

        # 쌍이 맞는 파일만 필터링
        print("파일 경로를 매칭 중입니다...")
        for s1_path in tqdm(self.s1_paths):
            # 폴더 구조: .../s1/ROIs..._s1_... -> .../s2/ROIs..._s2_...
            s2_path = s1_path.replace(f"{os.sep}s1{os.sep}", f"{os.sep}s2{os.sep}").replace("_s1_", "_s2_")
            
            if os.path.exists(s2_path):
                self.data_pairs.append({"s1": s1_path, "s2": s2_path})
        
        print(f"Dataset loaded: {len(self.data_pairs)} pairs found.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        paths = self.data_pairs[idx]
        
        # 1. Load SAR (2 Channels: VV, VH)
        with rasterio.open(paths['s1']) as src:
            s1_img = src.read().astype(np.float32) # (2, 256, 256) usually
            # nan 값 처리 (가끔 존재)
            s1_img = np.nan_to_num(s1_img, nan=-99.0)

        # 2. Load Optical (13 Channels)
        with rasterio.open(paths['s2']) as src:
            s2_img = src.read().astype(np.float32) # (13, 256, 256)

        # Tensor 변환
        return {
            'sar': torch.from_numpy(s1_img), 
            'opt': torch.from_numpy(s2_img)
        }

# -----------------------------------------------------------
# 2. 필터링 함수 (Refined Lee Filter)
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

# -----------------------------------------------------------
# 3. 유틸리티 함수 (히스토그램 역산)
# -----------------------------------------------------------
def get_quantile_from_hist(hist, bins, min_val, max_val, q):
    cdf = torch.cumsum(hist, dim=0)
    total_count = cdf[-1]
    target_rank = total_count * q
    idx = torch.searchsorted(cdf, target_rank)
    bin_width = (max_val - min_val) / bins
    value = min_val + (idx * bin_width)
    return value.item()

# -----------------------------------------------------------
# 4. 메인 계산 함수
# -----------------------------------------------------------
def calculate_sen12ms_stats_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"데이터셋 로딩: {SEN12MS_ROOT}")
    dataset = RawSEN12MSDataset(root_dir=SEN12MS_ROOT)

    # 샘플링
    num_samples = min(NUM_SAMPLES, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]
    subset_dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # --- 히스토그램 설정 ---
    # SAR 범위: -50dB ~ +50dB (충분히 넓게)
    SAR_MIN, SAR_MAX = -100.0, 50.0
    
    # Optical 범위: 0 ~ 10000 (Sentinel-2 L1C Raw Data Range)
    OPT_MIN, OPT_MAX = 0.0, 10000.0 
    
    NUM_BINS = 30000 
    
    hist_sar = torch.zeros(3, NUM_BINS).to(device) # (VV, VH, Diff)
    hist_opt = torch.zeros(3, NUM_BINS).to(device) # (R, G, B)
    
    print(f"Computing stats from {num_samples} samples using Histogram method...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 1. Data to GPU
            sar_batch = batch['sar'].to(device) # (B, 2, H, W)
            opt_batch = batch['opt'].to(device) # (B, 13, H, W)
            
            # --- [SAR Processing] ---
            # 이미 dB 단위라고 가정 (대부분의 GeoTIFF)
            # Lee Filter를 Linear 도메인에서 적용하려면 변환 필요
            sar_linear = 10 ** (sar_batch / 10.0)
            sar_filtered_linear = refined_lee_filter_pytorch(sar_linear) 
            sar_filtered_db = 10.0 * torch.log10(sar_filtered_linear + 1e-6)
            
            # 3채널 생성 (VV, VH, VV-VH)
            vv = sar_filtered_db[:, 0:1, :, :]
            vh = sar_filtered_db[:, 1:2, :, :]
            diff = vv - vh
            sar_final = torch.cat([vv, vh, diff], dim=1) # (B, 3, H, W)
            
            # --- [Optical Processing] ---
            # 13채널 중 RGB(Band 4, 3, 2) 추출
            # S2 밴드 순서: B1, B2, B3, B4... -> 인덱스 3(Red), 2(Green), 1(Blue)
            # 주의: 데이터셋마다 밴드 순서가 다를 수 있으나 일반적인 S2 .tif 기준
            opt_rgb = opt_batch[:, [3, 2, 1], :, :] # (B, 3, H, W)

            # --- 히스토그램 누적 ---
            for c in range(3):
                # SAR
                hist_sar[c] += torch.histc(
                    sar_final[:, c, :, :].float(), 
                    bins=NUM_BINS, 
                    min=SAR_MIN, 
                    max=SAR_MAX
                )
                # Optical
                hist_opt[c] += torch.histc(
                    opt_rgb[:, c, :, :].float(), 
                    bins=NUM_BINS, 
                    min=OPT_MIN, 
                    max=OPT_MAX
                )

    print("\n--- Calculation Complete. Computing Quantiles ---")
    
    # 3-Sigma에 해당하는 백분위수 (99.7% 커버 -> 양쪽 0.15% 제외)
    # 여기서는 좀 더 보수적인 Robust Scaling을 위해 0.1% ~ 99.9% 사용 (0.001, 0.999)
    # 또는 2% ~ 98%를 원하면 (0.02, 0.98) 사용
    
    q_low = 0.05 # 0.1%
    q_high = 0.95 # 99.9%

    sar_min_list, sar_max_list = [], []
    opt_min_list, opt_max_list = [], []
    
    channel_names_sar = ['VV', 'VH', 'Diff']
    channel_names_opt = ['Red', 'Green', 'Blue']

    print(f"\nTarget Percentiles: {q_low*100}% - {q_high*100}%")
    
    print("\n[SAR Stats (dB)]")
    for c in range(3):
        s_min = get_quantile_from_hist(hist_sar[c], NUM_BINS, SAR_MIN, SAR_MAX, q_low)
        s_max = get_quantile_from_hist(hist_sar[c], NUM_BINS, SAR_MIN, SAR_MAX, q_high)
        sar_min_list.append(s_min)
        sar_max_list.append(s_max)
        print(f"  {channel_names_sar[c]}: Min {s_min:.4f}, Max {s_max:.4f}")
        
    print("\n[Optical Stats (Raw Value 0-10000)]")
    for c in range(3):
        o_min = get_quantile_from_hist(hist_opt[c], NUM_BINS, OPT_MIN, OPT_MAX, q_low)
        o_max = get_quantile_from_hist(hist_opt[c], NUM_BINS, OPT_MIN, OPT_MAX, q_high)
        opt_min_list.append(o_min)
        opt_max_list.append(o_max)
        print(f"  {channel_names_opt[c]}: Min {o_min:.4f}, Max {o_max:.4f}")

    result = {
        "sar_min": sar_min_list,
        "sar_max": sar_max_list,
        "opt_min": opt_min_list,
        "opt_max": opt_max_list
    }
    
    print("\n[Copy & Paste Config]")
    print(f"SEN12MS_STATS = {result}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    calculate_sen12ms_stats_gpu()