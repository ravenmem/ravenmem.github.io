"""
Dataset classes and collate functions for SAR-to-Optical training.
"""

import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.transforms import make_transform
from utils.prompts import get_season_from_month, metadata_normalize

logger = logging.getLogger(__name__)


class SEN12_Dataset(Dataset):
    """
    Dataset for SEN12 format SAR/Optical image pairs.

    Expected directory structure:
        root_dir/ROIs{id}_{season}/s1_{idx}/image.png
        root_dir/ROIs{id}_{season}/s2_{idx}/image.png
    """

    def __init__(self, root_dir, transform: dict = None):
        """
        Args:
            root_dir: Dataset root path
            transform: Dict with 'sar' and 'opt' transform functions
        """
        self.root_dir = root_dir
        self.transform_dict = transform
        self.image_pairs = []  # (sar_path, opt_path, season) tuples

        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        logger.info(f"Scanning files in {root_dir}...")

        for root, dirs, files in os.walk(self.root_dir):
            # Extract season from path (e.g., ROIs1158_summer -> summer)
            season = "unknown"
            path_parts = root.split(os.sep)
            for part in path_parts:
                if part.startswith("ROIs") and "_" in part:
                    try:
                        season = part.split("_")[-1]
                    except IndexError:
                        pass
                    break

            for file in files:
                if file.lower().endswith(extensions):
                    s1_full_path = os.path.join(root, file)

                    if 's1_' in s1_full_path:
                        s2_full_path = s1_full_path.replace('s1_', 's2_')

                        if os.path.exists(s2_full_path):
                            self.image_pairs.append((s1_full_path, s2_full_path, season))

        self.image_pairs.sort()
        logger.info(f"Found {len(self.image_pairs)} pairs in {root_dir}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        sar_path, opt_path, season = self.image_pairs[idx]

        sar_image = Image.open(sar_path).convert('L')
        opt_image = Image.open(opt_path).convert('RGB')

        # Apply transforms
        if self.transform_dict and 'sar' in self.transform_dict:
            sar_tensor = self.transform_dict['sar'](sar_image)
        else:
            sar_tensor = make_transform(resize_size=128, data_type="sar", dataset="sen12")(sar_image)

        if self.transform_dict and 'opt' in self.transform_dict:
            opt_tensor = self.transform_dict['opt'](opt_image)
        else:
            opt_tensor = make_transform(resize_size=128, data_type="opt", dataset="sen12")(opt_image)

        return {
            "sar": sar_tensor,
            "opt": opt_tensor,
            "season": season,
            "input_ids": torch.tensor([0])
        }


def ben_collate_fn(batch):
    """
    Custom collate function for BENv2 dataset batches.

    Args:
        batch: List of (img_dict, labels, metadata_dict) tuples

    Returns:
        Tuple of (img_batch, label_batch, metadata_normalized, seasons)
    """
    imgs, labels, md_list = zip(*batch)

    sar_batch = torch.stack([b["sar"] for b in imgs], dim=0)
    opt_batch = torch.stack([b["opt"] for b in imgs], dim=0)
    img_batch = {"sar": sar_batch, "opt": opt_batch}

    label_batch = torch.stack([torch.as_tensor(l) for l in labels], dim=0)

    # Collect metadata
    md_raw = []
    seasons = []
    for md in md_list:
        md_raw.append([
            md["lon"], md["lat"], md["gsd"],
            md["cloud_cover"], md["year"], md["month"], md["day"]
        ])
        seasons.append(get_season_from_month(md["month"]))

    md_raw = torch.tensor(md_raw, dtype=torch.float32)
    md_norm = metadata_normalize(md_raw)

    return img_batch, label_batch, md_norm, seasons


def sen12ms_collate_fn(batch):
    """
    Custom collate function for SEN12MS dataset batches.

    SEN12MS doesn't have metadata like BENv2, so md_norm is None.
    Seasons are extracted from batch items if available.

    Args:
        batch: List of (img_dict, labels) tuples

    Returns:
        Tuple of (img_batch, label_batch, None, seasons)
    """
    imgs, labels = zip(*batch)

    sar_batch = torch.stack([b["sar"] for b in imgs], dim=0)
    opt_batch = torch.stack([b["opt"] for b in imgs], dim=0)
    img_batch = {"sar": sar_batch, "opt": opt_batch}

    label_batch = torch.stack([torch.as_tensor(l) for l in labels], dim=0)

    # Extract seasons from batch items if available
    seasons = [b.get("season", "") for b in imgs]

    return img_batch, label_batch, None, seasons
