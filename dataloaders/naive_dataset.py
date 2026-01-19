# dataloaders/sar_opt_dataset.py

import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SAR_OPT_Dataset(Dataset):
    def __init__(self, sar_folder, opt_folder, annotations_file, tokenizer, resolution=512, split='train', split_ratio=0.8, seed=42, null_text_ratio=0.5):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split = split
        self.sar_folder = sar_folder
        self.opt_folder = opt_folder
        self.null_text_ratio = null_text_ratio

        df = pd.read_csv(annotations_file)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * split_ratio)

        if split == 'train':
            self.annotations_df = df.iloc[:split_idx]
            print(f"'{annotations_file}'에서 훈련 데이터셋 {len(self.annotations_df)}개를 로드합니다.")
        elif split == 'val':
            self.annotations_df = df.iloc[split_idx:]
            print(f"'{annotations_file}'에서 검증 데이터셋 {len(self.annotations_df)}개를 로드합니다.")
        else:
            raise ValueError("split 인자는 'train' 또는 'val'이어야 합니다.")

            
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.vit_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):

        img_info = self.annotations_df.iloc[idx]
        relative_path = img_info['filepath']
        caption = img_info['caption']
        filename = relative_path.split('/')[-1]

        sar_image_path = os.path.join(self.sar_folder, relative_path)
        opt_image_path = os.path.join(self.opt_folder, filename)
        
        try:
            sar_image = Image.open(sar_image_path).convert("RGB")
            opt_image = Image.open(opt_image_path).convert("RGB")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. SAR 경로: {sar_image_path} 또는 OPT 경로: {opt_image_path}")
            return None

        if self.split == 'train' and random.random() < self.null_text_ratio:
            caption = ""
            
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        opt_pixel_values = self.image_transforms(opt_image)
        sar_control_values = self.image_transforms(sar_image)
        sar_vit_values = self.vit_transforms(sar_image)

        return {
            "opt_pixel_values": opt_pixel_values,
            "conditioning_pixel_values": sar_control_values,
            "representation_values": sar_vit_values,
            "input_ids": input_ids.squeeze(0)
        }