"""
Prompt generation utilities and class/season constants for SAR-to-Optical synthesis.
"""

import math
import random
import torch


# BigEarthNet v2 class names
BEN_V2_CLASS_NAMES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric"
]

# Season mapping
SEASON_MAP = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}

# Stable Diffusion 2.1 compatible class prompts for BENv2
SD_V2_CLASS_PROMPTS = [
    "farmland with trees",                  # 0: Agro-forestry areas
    "crop fields",                          # 1: Arable land
    "sandy beach and dunes",                # 2: Beaches, dunes, sands
    "deciduous broadleaf forest",           # 3: Broad-leaved forest
    "coastal marshland",                    # 4: Coastal wetlands
    "patchwork farmland",                   # 5: Complex cultivation patterns
    "pine forest",                          # 6: Coniferous forest
    "industrial district and warehouses",   # 7: Industrial or commercial units
    "lakes and rivers",                     # 8: Inland waters
    "swamp and marshland",                  # 9: Inland wetlands
    "farmland with wild vegetation",        # 10: Land principally occupied by agriculture...
    "open ocean",                           # 11: Marine waters
    "mixed forest",                         # 12: Mixed forest
    "heathland and dry scrub",              # 13: Moors, heathland and sclerophyllous vegetation
    "wild grassland and rocky ground",      # 14: Natural grassland and sparsely vegetated areas
    "green pasture",                        # 15: Pastures
    "vineyards and orchards",               # 16: Permanent crops
    "young woodland and shrubs",            # 17: Transitional woodland, shrub
    "dense city buildings"                  # 18: Urban fabric
]

# SEN12MS LCCS Land Use class prompts (11 classes)
LCCS_LU_CLASS_PROMPTS_V2 = [
    "dense natural forest",          # 10
    "open sparse forest",            # 20
    "natural grassland",             # 30
    "woody shrubland",               # 40
    "herbaceous cropland",           # 36
    "dense urban area",              # 9
    "barren rocky land",             # 1
    "permanent snow and ice",        # 2
    "inland water body",             # 3
    "forest and cropland mosaic",    # 25
    "grassland and cropland mosaic", # 35
]


def get_season_from_month(month):
    """
    Convert month (1-12) to season string.

    Args:
        month: Integer 1-12

    Returns:
        str: Season name ("Spring", "Summer", "Autumn", "Winter")
    """
    m = int(month)
    if 3 <= m <= 5:
        return "Spring"
    elif 6 <= m <= 8:
        return "Summer"
    elif 9 <= m <= 11:
        return "Autumn"
    else:
        return "Winter"  # 12, 1, 2


def metadata_normalize(
    metadata: torch.Tensor,
    lon_min=-180., lon_max=180.,
    lat_min=-90., lat_max=90.,
    year_min=1980., year_max=2100.,
):
    """
    Normalize geospatial metadata tensor.

    Args:
        metadata: Tensor of shape (B, 7) with [lon, lat, gsd, cloud_cover, year, month, day]

    Returns:
        Tensor of shape (B, 2) with [month_sin, month_cos] for cyclic encoding
    """
    lon, lat, gsd, cloud_cover, year, month, day = metadata.unbind(dim=-1)

    # Cyclic encoding for month (1-12)
    month_rad = 2 * math.pi * (month - 1) / 12.0
    month_sin = torch.sin(month_rad)
    month_cos = torch.cos(month_rad)

    return torch.stack([month_sin, month_cos], dim=-1)


def logits_to_prompt(args, is_train, logits, class_names, seasons=None, threshold=0.5, max_classes=3):
    """
    Convert classifier logits to text prompts for diffusion model.

    Args:
        args: Argument namespace with null_text_ratio and fixed_prompt
        is_train: Whether in training mode (enables augmentation)
        logits: Classifier output tensor (B, num_classes)
        class_names: List of class name strings
        seasons: Optional list of season strings per batch item
        threshold: Probability threshold for class activation
        max_classes: Maximum number of classes in prompt

    Returns:
        List of prompt strings
    """
    TEMPLATES = [
        "Electro-Optical Image of {} in {}",
    ]
    DEFAULT_TEMPLATES = [
        "Electro-Optical Image of {}",
    ]

    probs = torch.sigmoid(logits)
    prompts = []

    for i in range(probs.shape[0]):
        curr_probs = probs[i]
        current_season = seasons[i] if seasons is not None else ""

        # Find classes above threshold
        active_indices = torch.where(curr_probs > threshold)[0]

        class_items = []
        if len(active_indices) > 0:
            # Sort by probability (descending)
            selected_probs = curr_probs[active_indices]
            sorted_order = torch.argsort(selected_probs, descending=True)
            sorted_indices = active_indices[sorted_order]
            final_indices = sorted_indices[:max_classes]

            class_items = [class_names[idx.item()] for idx in final_indices]

            # Shuffle during training for augmentation
            if is_train:
                random.shuffle(class_items)

            cls_str = ', '.join(class_items)
        else:
            cls_str = "Satellite landscape"

        # Build prompt with template
        if current_season:
            if is_train:
                template = random.choice(TEMPLATES)
            else:
                template = TEMPLATES[0]
            prompt = template.format(cls_str, current_season)
        else:
            template = random.choice(DEFAULT_TEMPLATES) if is_train else DEFAULT_TEMPLATES[0]
            prompt = template.format(cls_str)

        # Fixed prompt override
        if args.fixed_prompt:
            prompt = args.fixed_prompt

        # Null text drop (Classifier-Free Guidance)
        if is_train and random.random() < args.null_text_ratio:
            prompt = ""

        prompts.append(prompt)

    return prompts
