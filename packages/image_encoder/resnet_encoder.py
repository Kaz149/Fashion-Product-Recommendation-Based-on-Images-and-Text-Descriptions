import os
import sys
sys.path.insert(0, os.getcwd())
try:
    from pipeline._env_setup import setup_env
    setup_env()
except Exception:
    pass
from typing import List, Union, Optional
from pathlib import Path
import yaml
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ResNetEncoder:
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        cfg = load_config()
        rc = cfg["image_encoder"]["resnet"]
        if embedding_dim is None:
            embedding_dim = rc["embedding_dim"]
        if pretrained is None:
            pretrained = bool(rc["pretrained"])
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None
        self._transform = None
        self._loaded = False

    def _build_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._transform = self._build_transform()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
        full = models.resnet50(weights=weights)
        children = list(full.children())[:-1]
        self._model = torch.nn.Sequential(*children).to(self.device).eval()
        self._loaded = True

    def _to_images(self, inputs: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
        loaded = []
        for x in inputs:
            if isinstance(x, (str, Path)):
                loaded.append(Image.open(str(x)).convert("RGB"))
            else:
                loaded.append(x)
        return loaded

    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        self._ensure_loaded()
        imgs = self._to_images(images)
        tensors = torch.stack([self._transform(img) for img in imgs], dim=0).to(self.device)
        with torch.no_grad():
            feat = self._model(tensors)
        feat = feat.squeeze(-1).squeeze(-1)
        feats = feat.cpu().numpy().astype(np.float32)
        if feats.shape[1] != self.embedding_dim:
            raise ValueError(f"ResNet50 expected dim {self.embedding_dim}, got {feats.shape[1]}")
        return feats

    def __call__(self, images):
        if isinstance(images, list):
            return self.encode_images(images)
        return self.encode_image(images)
