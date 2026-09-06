import os
import yaml
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_transform(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augment: bool = False,
) -> transforms.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image(image_path: Union[str, Path]) -> Image.Image:
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img


def preprocess_image(
    image_path: Union[str, Path],
    image_size: int = 224,
    augment: bool = False,
    return_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    img = load_image(image_path)
    tf = get_image_transform(image_size=image_size, augment=augment)
    tensor = tf(img)
    if return_numpy:
        return tensor.permute(1, 2, 0).cpu().numpy()
    return tensor


def preprocess_image_batch(
    image_paths: List[Union[str, Path]],
    image_size: int = 224,
    augment: bool = False,
) -> torch.Tensor:
    tensors = [preprocess_image(p, image_size=image_size, augment=augment) for p in image_paths]
    return torch.stack(tensors, dim=0)


def resize_and_save(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    image_size: int = 224,
    overwrite: bool = False,
) -> bool:
    if os.path.isfile(dst_path) and not overwrite:
        return False
    try:
        img = load_image(src_path)
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)
        return True
    except Exception:
        return False
