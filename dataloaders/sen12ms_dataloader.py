import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import random  # [NEW] 랜덤 분할을 위해 추가

import torch
from torch.utils.data import Dataset, DataLoader
from utils.transforms import make_transform

class SEN12MSDataset(Dataset):
    def __init__(self, root_dir, subset="train", transform=None, seed=42):
        """
        Args:
            root_dir (str): 데이터셋 루트 경로
            subset (str): 'train' (80%), 'test' (20%) 중 하나.
            transform (callable, optional): 추가적인 사용자 정의 변환
            seed (int): 데이터 분할을 위한 고정 시드값 (기본값 42)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        
        # 유효성 검사
        assert self.subset in ["train", "test"], "subset은 'train' 또는 'test'여야 합니다."

        # -------------------------------------------------------------------
        # LCCS LU Class Definition
        # -------------------------------------------------------------------
        self.lccs_lu_values = [10, 20, 30, 40, 36, 9, 25, 35, 2, 1, 3]
        self.num_classes = len(self.lccs_lu_values)
        self.val_to_idx = {v: i for i, v in enumerate(self.lccs_lu_values)}

        # -------------------------------------------------------------------
        # 1. 모든 파일 경로 로딩 및 쌍(Pair) 구성
        # -------------------------------------------------------------------
        s1_paths = glob.glob(os.path.join(root_dir, "*", "s1", "*.tif"))
        all_data_pairs = []

        for s1_path in s1_paths:
            # 경로 문자열 치환을 통해 S2와 LC 경로 유추
            # OS에 따라 separator가 다를 수 있으므로 주의
            s2_path = s1_path.replace(f"{os.sep}s1{os.sep}", f"{os.sep}s2{os.sep}").replace("_s1_", "_s2_")
            lc_path = s1_path.replace(f"{os.sep}s1{os.sep}", f"{os.sep}lc{os.sep}").replace("_s1_", "_lc_")
            
            if os.path.exists(s2_path) and os.path.exists(lc_path):
                all_data_pairs.append({
                    "s1": s1_path,
                    "s2": s2_path,
                    "lc": lc_path
                })
        
        # -------------------------------------------------------------------
        # 2. Train / Test Split (8:2, Seed 고정)
        # -------------------------------------------------------------------
        # (중요) OS별로 glob 읽는 순서가 다를 수 있으므로, 셔플 전에 반드시 정렬해야 함
        all_data_pairs.sort(key=lambda x: x['s1'])
        
        # 랜덤 셔플 (Global seed에 영향주지 않기 위해 로컬 인스턴스 사용 권장)
        rng = random.Random(seed)
        rng.shuffle(all_data_pairs)
        
        # 분할 지점 계산 (80%)
        split_idx = int(len(all_data_pairs) * 0.8)
        
        if self.subset == "train":
            self.data_pairs = all_data_pairs[:split_idx]
        else: # subset == "test"
            self.data_pairs = all_data_pairs[split_idx:]
        
        print(f"[{self.subset.upper()}] Dataset loaded.")
        print(f"Total found: {len(all_data_pairs)} -> Allocated: {len(self.data_pairs)}")

        self.pos_weight = self._compute_class_weights()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        paths = self.data_pairs[idx]
        
        # 1. Load Data (Numpy)
        # rasterio는 스레드 안전하지 않을 수 있으므로 주의 (멀티워커 사용시)
        with rasterio.open(paths['s1']) as src:
            s1_img = src.read().astype(np.float32) # (C, H, W)
            
        with rasterio.open(paths['s2']) as src:
            s2_img = src.read().astype(np.float32) # (C, H, W)
            
        with rasterio.open(paths['lc']) as src:
            lc_img = src.read(3).astype(np.int64)  # (H, W) - IGBP Scheme 등 layer 확인 필요

        # -------------------------------------------------------------------
        # Generate Label Vector
        # -------------------------------------------------------------------
        unique_vals, counts = np.unique(lc_img, return_counts=True)
        total_pixels = lc_img.size
        threshold_count = total_pixels * 0.05
        
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for val, count in zip(unique_vals, counts):
            if count >= threshold_count:
                if val in self.val_to_idx:
                    idx_class = self.val_to_idx[val]
                    label_vec[idx_class] = 1.0 

        # 2. To Tensor
        s1_tensor = torch.from_numpy(s1_img)
        s2_tensor = torch.from_numpy(s2_img)
        lc_tensor = torch.from_numpy(lc_img) 

        # 3. Pack
        sample = {
            's1': s1_tensor, 
            's2': s2_tensor, 
            'lc': lc_tensor, 
            'label': label_vec
        }
        sar_channels = torch.Tensor(sample['s1'][[0, 1], :, :])
        opt_channels = torch.Tensor(sample['s2'][[3, 2, 1], :, :])


        # 4. Transform (Train set인 경우 증강 적용 등)
        if self.transform:

            sar = self.transform["sar"]["preprocess"](sar_channels)
            opt = self.transform["opt"]["preprocess"](opt_channels)
            
            combined = torch.cat([sar, opt], dim=0)

            shared_augment = self.transform["opt"]["augment"]
            augmented_combined = shared_augment(combined)
            
            # print('augmented_combine.shape', augmented_combined.shape)
            
            if augmented_combined.shape[0] == 5:
                sar = augmented_combined[:2, :, :]
                opt = augmented_combined[2:, :, :]
            else:
                sar = augmented_combined[:3, :, :]
                opt = augmented_combined[3:, :, :]

            # print('sar.shape after augment:', sar.shape)
            # print('opt.shape after augment:', opt.shape)
            
            
            sar = self.transform["sar"]["normalize"](sar)
            opt = self.transform["opt"]["normalize"](opt)

            img = {"sar": sar, "opt": opt}
        else:
            img = torch.from_numpy(img)

        return img, sample['label']

    def _compute_class_weights(self, max_weight=10.0):
        """
        Args:
            max_weight (float): 계산된 가중치가 이 값보다 크면 이 값으로 자릅니다 (Clipping).
                                None일 경우 제한하지 않습니다.
        """
        print(f"\n[{self.subset.upper()}] Computing class weights... (Scanning {len(self.data_pairs)} files)")
        
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        total_valid_samples = 0
        
        # 1. 전체 데이터 순회 (이전과 동일)
        for idx, paths in enumerate(self.data_pairs):
            try:
                with rasterio.open(paths['lc']) as src:
                    lc_img = src.read(3).astype(np.int64)
                
                unique_vals, counts = np.unique(lc_img, return_counts=True)
                threshold = lc_img.size * 0.05 # 라벨 생성 하한선 (5%)
                
                has_valid_class = False
                for val, count in zip(unique_vals, counts):
                    if count >= threshold:
                        if val in self.val_to_idx:
                            class_idx = self.val_to_idx[val]
                            class_counts[class_idx] += 1
                            has_valid_class = True
                
                if has_valid_class:
                    total_valid_samples += 1

            except Exception as e:
                pass # 에러 처리 생략

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(self.data_pairs)}...")

        print(f"Done. Valid Samples: {total_valid_samples}")
        print(f"Class Counts: {class_counts}")

        # 2. 가중치 계산 및 상한 적용 (Clipping)
        weights = []
        for count in class_counts:
            if count > 0:
                # Balanced Weight Formula
                w = total_valid_samples / (self.num_classes * count)
                
                # [NEW] 상한 적용
                if max_weight is not None:
                    if w > max_weight:
                        print(f"  -> Clipping weight {w:.2f} to {max_weight}")
                        w = max_weight
            else:
                w = 1.0 # 데이터가 없는 클래스 기본값
                
            weights.append(w)
            
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return weights_tensor
    
    def get_class_weights(self):
        """External method to retrieve calculated weights"""
        return self.pos_weight






# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset_root = "./sen12ms"
    RESIZE_SIZE = 256
    DATA_TYPE="opt"

    transform = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=True, train_datatype="opt", dataset="sen12ms", calc_norm=True), # opt
        "sar": make_transform(
            resize_size=RESIZE_SIZE, 
            data_type="sar", 
            is_train=True,
            train_datatype="sar", # sar
            dataset="sen12ms",
            calc_norm=True
        )
    }

    transform_val = {
        "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False, train_datatype="opt", dataset="sen12ms", calc_norm=True), # opt
        "sar": make_transform(
            resize_size=RESIZE_SIZE,
            data_type="sar", 
            is_train=False,
            train_datatype="sar", # sar
            dataset="sen12ms",
            calc_norm=True
        )
    }
    
    if os.path.exists(dataset_root):
        print(">>> Loading TRAIN dataset...")
        train_ds = SEN12MSDataset(root_dir=dataset_root, subset="train", seed=42, transform=transform)
        
        print("\n>>> Loading TEST dataset...")
        test_ds = SEN12MSDataset(root_dir=dataset_root, subset="test", seed=42, transform=transform_val)
        
        if len(train_ds) > 0:
            image, label = test_ds[283]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            print(image['sar'].shape)
            print(image['opt'].shape)
            print(image['opt'].min(), image['opt'].max())
            print(image['sar'].min(), image['sar'].max())
            # assert False

            sar_viz = image['sar'][0, :, :].numpy()
            axes[0].imshow(sar_viz, cmap='gray')
            axes[0].set_title("Transformed SAR Channel 1 (VV, VH, VV-VH)")
            opt_viz = image['opt'].permute(1, 2, 0).numpy()
            axes[1].imshow(opt_viz)
            axes[1].set_title("Transformed Optical RGB")
            plt.suptitle(f"Label Vector: {label.numpy()}")
            plt.tight_layout()
            plt.savefig("transformed_sample.png")
            print(label)
    else:
        print(f"경로를 찾을 수 없습니다: {dataset_root}")