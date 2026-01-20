from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import random

# BEN_STATS = {
#     # SAR: [VV, VH, VV-VH] 순서
#     # "sar_min": [-27.525001525878906, -31.81500244140625, 0.07999420166015625],
#     # "sar_max": [-5.1100006103515625, -11.319999694824219, 14.669998168945312],
#     "sar_min": [-33.0, -35.36499786376953, -3.75],
#     "sar_max": [2.029998779296875, -5.855003356933594, 19.334999084472656],
    
#     # OPT: [R, G, B] 순서 (BigEarthNet B04, B03, B02 대응 가정)
#     # "opt_min": [27.0, 62.66666793823242, 46.66666793823242],
#     # "opt_max": [1974.3333740234375, 1539.666748046875, 1160.0]

#     ### gamma 분포 변환 후
#     "opt_min": [1.0, 1.0, 1.0],
#     "opt_max": [71.0, 70.66667175292969, 71.0]
# }

# BEN_STATS = {
#     # SAR: [VV, VH, VV-VH] 순서
#     "sar_min": [-27.529998779296875, -31.800003051757812, 0.06499481201171875],
#     "sar_max": [-5.1300048828125, -11.325004577636719, 14.659996032714844],
    
#     # OPT: [R, G, B] 순서 (BigEarthNet B04, B03, B02 대응 가정)
#     "opt_min": [26.666667938232422, 62.333335876464844, 47.0],
#     "opt_max": [1973.0, 1537.0, 1159.3333740234375]
# }


BEN_STATS = {
    # SAR: [VV, VH, VV-VH] 순서
    # "sar_min": [-27.525001525878906, -31.81500244140625, 0.07999420166015625],
    # "sar_max": [-5.1100006103515625, -11.319999694824219, 14.669998168945312],
    "sar_min": [-33.0, -35.36499786376953, -3.75],
    "sar_max": [2.029998779296875, -5.855003356933594, 19.334999084472656],
    
    # OPT: [R, G, B] 순서 (BigEarthNet B04, B03, B02 대응 가정)
    "opt_min": [27.0, 62.66666793823242, 46.66666793823242],
    "opt_max": [1974.3333740234375, 1539.666748046875, 1160.0]
}

SEN12MS_STATS = {
    # SAR: [VV, VH, VV-VH] 순서
    # "sar_min": [-27.525001525878906, -31.81500244140625, 0.07999420166015625],
    # "sar_max": [-5.1100006103515625, -11.319999694824219, 14.669998168945312],

    ### 1.5sigma
    # "sar_min": [-17.81000518798828, -26.279998779296875, 3.2949981689453125],
    # "sar_max": [-6.720001220703125, -13.135002136230469, 10.860000610351562],


    ### 2sigma
    "sar_min": [-20.029998779296875, -28.540000915527344, 2.3549957275390625],
    "sar_max": [-5.1150054931640625, -11.849998474121094, 12.264999389648438],
    
    # OPT: [R, G, B] 순서 (BigEarthNet B04, B03, B02 대응 가정)
    # "opt_min": [267.0, 486.0, 703.0],
    # "opt_max": [4293.0, 3141.0, 2743.0]
    
    ### 1.5sigma
    # "opt_min": [425.0, 679.0, 829.0],
    # "opt_max": [1989.0, 1627.0, 1587.0]


    ### 2sigma
    "opt_min": [362.0, 607.0, 783.0],
    "opt_max": [2675.0, 2031.0, 1825.0]
    
    # "opt_min": [26.0, 49.0, 88.0],
    # "opt_max": [9241.0, 8597.0, 9428.0]
}


def refined_lee_filter_pytorch(img_tensor, kernel_size=3, k=6):
    """
    Refined Lee Filter (RLF) - GPU Optimized
    Input tensor must be on GPU for acceleration.
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

    # 1. Unfold (GPU에서 병렬 처리됨)
    patches = F.unfold(img_in, kernel_size=kernel_size, padding=pad)
    patches = patches.view(img_in.size(0), img_in.size(1), kernel_size*kernel_size, -1)
    
    # 2. 중심 픽셀
    center_idx = (kernel_size * kernel_size) // 2
    center_pixels = patches[:, :, center_idx, :].unsqueeze(2)

    # 3. 거리 계산 및 Top-K (GPU 가속 핵심 부분)
    distances = torch.abs(patches - center_pixels)
    _, topk_indices = torch.topk(distances, k, dim=2, largest=False, sorted=False)

    # 4. Gather
    knn_patches = torch.gather(patches, 2, topk_indices)

    # 5. 통계량 계산
    rlf_local_mean = torch.mean(knn_patches, dim=2)
    rlf_local_var = torch.var(knn_patches, dim=2)

    # 6. 필터링
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


def lee_filter_pytorch(img_tensor, kernel_size=5):
    pad = kernel_size // 2
    
    if img_tensor.dim() == 3:
        img_in = img_tensor.unsqueeze(0)
    else:
        img_in = img_tensor

    local_mean = F.avg_pool2d(img_in, kernel_size, stride=1, padding=pad, count_include_pad=False)
    
    local_sqr_mean = F.avg_pool2d(img_in**2, kernel_size, stride=1, padding=pad, count_include_pad=False)
    local_var = local_sqr_mean - local_mean**2
    

    overall_var = local_var.mean() 

    epsilon = 1e-6
    k = local_var / (local_var + overall_var + epsilon)
    
    result = local_mean + k * (img_in - local_mean)
    
    if img_tensor.dim() == 3:
        return result.squeeze(0)
    return result

def append_diff_channel(img_tensor):
    """
    입력: (2, H, W) 형태의 SAR 텐서 (0: VV, 1: VH)
    출력: (3, H, W) 형태의 텐서 (0: VV, 1: VH, 2: VV-VH)
    """
    # 채널 차원(dim=0) 분리
    vv = img_tensor[0:1, :, :]
    vh = img_tensor[1:2, :, :]
    
    # 세 번째 채널 생성 (Difference)
    diff = vv - vh
    
    # 3채널로 합치기
    return torch.cat([vv, vh, diff], dim=0)

def robust_normalize_with_stats(img_tensor, min_vals, max_vals):
    """
    미리 계산된 min/max 통계량을 사용하여 0~1로 정규화 (Clipping & Scaling)
    Input: (C, H, W) Tensor
    """
    # 디바이스 및 타입 일치 확인
    device = img_tensor.device
    dtype = img_tensor.dtype
    
    # 리스트를 텐서로 변환 후 브로드캐스팅 shape (C, 1, 1)로 변경
    # clone().detach()는 안전한 텐서 생성을 위함
    mins = torch.tensor(min_vals, device=device, dtype=dtype).view(-1, 1, 1)
    maxs = torch.tensor(max_vals, device=device, dtype=dtype).view(-1, 1, 1)
    
    # 1. Clipping (Outlier 제거)
    # min 값보다 작은건 min으로, max 값보다 큰건 max로
    img_tensor = torch.clamp(img_tensor, min=mins, max=maxs)
    
    # 2. Min-Max Scaling (0 ~ 1)
    denominator = maxs - mins + 1e-6
    img_tensor = (img_tensor - mins) / denominator
    
    return img_tensor

def make_transform(is_train: bool = True, data_type: str = "opt", dataset: str = "benv2", 
                   resize_size: int = 256, calc_norm: bool = False, train_datatype="opt"):
    # NORM_MEAN = (0.485, 0.456, 0.406)
    # NORM_STD = (0.229, 0.224, 0.225)

    NORM_MEAN=(0.430, 0.411, 0.296)
    NORM_STD=(0.213, 0.156, 0.143)
        
    preprocess_list = [
        v2.ToTensor(),
        v2.Resize((resize_size, resize_size), antialias=True),
    ]

    if data_type == "opt":
        if dataset == "benv2" and train_datatype=="opt":
            # preprocess_list.append(v2.Lambda(lambda x: torch.pow(x, 0.5)))
            preprocess_list.append(
                v2.Lambda(lambda x: robust_normalize_with_stats(
                    x, 
                    min_vals=BEN_STATS["opt_min"], 
                    max_vals=BEN_STATS["opt_max"]
                ))
            )

        if dataset == "sen12ms" and train_datatype=="opt":
            preprocess_list.append(
                v2.Lambda(lambda x: robust_normalize_with_stats(
                    x, 
                    min_vals=SEN12MS_STATS["opt_min"], 
                    max_vals=SEN12MS_STATS["opt_max"]
                ))
            )


    elif data_type == "sar":
        if dataset == "benv2" and train_datatype=="sar":
            def apply_filter_in_linear_domain(x_db):
                x_linear = 10 ** (x_db / 10.0)
                
                x_filtered_linear = refined_lee_filter_pytorch(x_linear, kernel_size=3, k=6)
                
                x_filtered_db = 10.0 * torch.log10(x_filtered_linear + 1e-6)
                
                return x_filtered_db

            preprocess_list.append(v2.Lambda(apply_filter_in_linear_domain))
            preprocess_list.append(v2.Lambda(append_diff_channel))
            preprocess_list.append(
                v2.Lambda(lambda x: robust_normalize_with_stats(
                    x, 
                    min_vals=BEN_STATS["sar_min"], 
                    max_vals=BEN_STATS["sar_max"]
                ))
            )
        if dataset == "sen12ms" and train_datatype=="sar":
            def apply_filter_in_linear_domain(x_db):
                x_linear = 10 ** (x_db / 10.0)
                x_filtered_linear = refined_lee_filter_pytorch(x_linear, kernel_size=3, k=6)
                x_filtered_db = 10.0 * torch.log10(x_filtered_linear + 1e-6)
                return x_filtered_db

            preprocess_list.append(v2.Lambda(apply_filter_in_linear_domain))
            preprocess_list.append(v2.Lambda(append_diff_channel))
            preprocess_list.append(
                v2.Lambda(lambda x: robust_normalize_with_stats(
                    x, 
                    min_vals=SEN12MS_STATS["sar_min"], 
                    max_vals=SEN12MS_STATS["sar_max"]
                ))
            )


    augment_list = []
    if is_train:
        augment_list.extend([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
    
    normalize_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=NORM_MEAN, std=NORM_STD) if not calc_norm else v2.Identity()
    ])


    augment_transform = v2.Compose(augment_list) if augment_list else v2.Identity()

    return {
        "preprocess": v2.Compose(preprocess_list),
        "augment": augment_transform,
        "normalize": normalize_transform
    }

