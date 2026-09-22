from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import torchvision
try:
    import open_clip
except ImportError:
    open_clip = None
from tqdm import tqdm
from config import PATHS, IMAGE_MODELS, CLIP, DEVICE
try:
    from src.scratch_cnn_model import load_backbone_for_embedding
except ImportError:
    load_backbone_for_embedding = None
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class BaseImageEmbedder(ABC):
    dim: int
    device: str = DEVICE
    transform: Callable

    def __init_subclass__(cls, dim: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.dim = dim

    @abstractmethod
    def _embed_batch_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        ...

    def _to_tensor(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed(self, img_path: Union[Path, str]) -> np.ndarray:
        img_path = Path(img_path)
        tensor = self._to_tensor(img_path)
        batch_emb = self._embed_batch_tensor(tensor)
        v = batch_emb[0]
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return (v / norm).astype(np.float32)

    @torch.no_grad()
    def embed_batch(self, img_paths: List[Union[Path, str]], batch_size: int=32, verbose: bool=True) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        iterator = range(0, len(img_paths), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc='Embedding images')
        for start_idx in iterator:
            batch_paths = img_paths[start_idx:start_idx + batch_size]
            tensors = []
            for p in batch_paths:
                tensors.append(self._to_tensor(Path(p)))
            batch_tensor = torch.cat(tensors, dim=0)
            batch_emb = self._embed_batch_tensor(batch_tensor)
            all_embeddings.append(batch_emb)
        if not all_embeddings:
            return np.zeros((0, self.dim), dtype=np.float32)
        stacked = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (stacked / norms).astype(np.float32)

class ScratchCNNEmbedder(BaseImageEmbedder, dim=IMAGE_MODELS.DIMS['scratch']):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __init__(self, ckpt_path: Path=PATHS.SCRATCH_CNN_CKPT):
        super().__init__()
        if load_backbone_for_embedding is None:
            raise ImportError('Không thể import load_backbone_for_embedding từ src.scratch_cnn_model. Vui lòng đảm bảo file src/scratch_cnn_model.py tồn tại và định nghĩa hàm này.')
        self.model, _ = load_backbone_for_embedding(ckpt_path, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _embed_batch_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        features = self.model.forward_features(batch_tensor)
        return features.detach().cpu().numpy().astype(np.float32)

class ResNet50Embedder(BaseImageEmbedder, dim=IMAGE_MODELS.DIMS['resnet']):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        base = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())
        self.backbone = self.backbone.to(self.device)
        self.pool = self.pool.to(self.device)
        self.backbone.eval()
        self.pool.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for param in self.pool.parameters():
            param.requires_grad_(False)

    def _embed_batch_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        features = self.backbone(batch_tensor)
        pooled = self.pool(features)
        return pooled.detach().cpu().numpy().astype(np.float32)

class EfficientNetB0Embedder(BaseImageEmbedder, dim=IMAGE_MODELS.DIMS['effnet']):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __init__(self):
        super().__init__()
        base = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())
        self.backbone = self.backbone.to(self.device)
        self.pool = self.pool.to(self.device)
        self.backbone.eval()
        self.pool.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for param in self.pool.parameters():
            param.requires_grad_(False)

    def _embed_batch_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        features = self.backbone(batch_tensor)
        pooled = self.pool(features)
        return pooled.detach().cpu().numpy().astype(np.float32)

class CLIPImageEmbedder(BaseImageEmbedder, dim=IMAGE_MODELS.DIMS['clip']):

    def __init__(self):
        super().__init__()
        if open_clip is None:
            raise ImportError('open_clip chưa được cài đặt. Vui lòng chạy: pip install open-clip-torch')
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(CLIP.MODEL_NAME, pretrained=CLIP.PRETRAINED, device=DEVICE)
        self.transform = self.preprocess_val
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _embed_batch_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        encoded = self.model.encode_image(batch_tensor)
        return encoded.float().detach().cpu().numpy().astype(np.float32)

def get_image_embedder(name: str, **kwargs) -> BaseImageEmbedder:
    if name == 'scratch':
        return ScratchCNNEmbedder(**kwargs)
    elif name == 'resnet':
        return ResNet50Embedder(**kwargs)
    elif name == 'effnet':
        return EfficientNetB0Embedder(**kwargs)
    elif name == 'clip':
        return CLIPImageEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown image embedder: '{name}'. Available: {list(IMAGE_MODELS.AVAILABLE)}")
