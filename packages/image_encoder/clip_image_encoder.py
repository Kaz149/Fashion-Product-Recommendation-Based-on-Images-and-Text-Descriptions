import os
from typing import List, Union, Optional
from pathlib import Path
import yaml
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import CLIPModel as _CLIPModelHF


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CLIPImageEncoder:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = 512,
    ):
        cfg = load_config()
        if model_name is None:
            model_name = cfg["image_encoder"]["clip_image"]["model_name"]
            embedding_dim = cfg["image_encoder"]["clip_image"]["embedding_dim"]
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._processor = None
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
        self._model.eval().to(self.device)
        self._loaded = True

    @staticmethod
    def _unwrap(out):
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "image_embeds") and getattr(out, "image_embeds") is not None:
            return out.image_embeds
        if hasattr(out, "text_embeds") and getattr(out, "text_embeds") is not None:
            return out.text_embeds
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output") is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and getattr(out, "last_hidden_state") is not None:
            return out.last_hidden_state[:, 0, :]
        # fallback 
        return out[0]

    def _preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        self._ensure_loaded()
        proc = self._processor(images=images, return_tensors="pt")
        return proc["pixel_values"].to(self.device)

    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert("RGB")
        else:
            img = image
        return self.encode_images([img])[0]

    def encode_images(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        loaded = []
        for x in images:
            if isinstance(x, (str, Path)):
                loaded.append(Image.open(str(x)).convert("RGB"))
            else:
                loaded.append(x)
        pixels = self._preprocess(loaded)
        with torch.no_grad():
            out = self._model(pixel_values=pixels)
            emb = self._unwrap(out)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)

    def __call__(self, images):
        if isinstance(images, list):
            return self.encode_images(images)
        return self.encode_image(images)
