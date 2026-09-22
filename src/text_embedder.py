from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import open_clip
    import torch
except ImportError:
    open_clip = None
    torch = None
from config import PATHS, TEXT_MODELS, CLIP, DEVICE

class BaseTextEmbedder(ABC):
    dim: int

    def __init_subclass__(cls, dim: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.dim = dim

    def fit(self, corpus: List[str]) -> None:
        pass

    @abstractmethod
    def _embed(self, texts: List[str]) -> np.ndarray:
        ...

    def embed(self, text: str) -> np.ndarray:
        batch_emb = self._embed([text])
        v = batch_emb[0]
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return (v / norm).astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int=64, verbose: bool=True) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc='Embedding texts')
        for start_idx in iterator:
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_emb = self._embed(batch_texts)
            all_embeddings.append(batch_emb)
        if not all_embeddings:
            return np.zeros((0, self.dim), dtype=np.float32)
        stacked = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (stacked / norms).astype(np.float32)

class TFIDFEmbedder(BaseTextEmbedder, dim=TEXT_MODELS.DIMS['tfidf']):

    def __init__(self, max_features: int=TEXT_MODELS.TFIDF_MAX_FEATURES, ngram_range: tuple=TEXT_MODELS.TFIDF_NGRAM, min_df: int=TEXT_MODELS.TFIDF_MIN_DF, stop_words: str=TEXT_MODELS.STOP_WORDS):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.stop_words = stop_words
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, stop_words=stop_words)
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self.vec.fit_transform(corpus)
        self._fitted = True
        PATHS.TFIDF_VECTORIZER.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, PATHS.TFIDF_VECTORIZER)
        print(f'TF-IDF fitted vocab len={len(self.vec.vocabulary_)}')

    @classmethod
    def load_fitted(cls) -> 'TFIDFEmbedder':
        if not PATHS.TFIDF_VECTORIZER.exists():
            raise FileNotFoundError(f'Chưa tìm thấy TF-IDF vectorizer tại: {PATHS.TFIDF_VECTORIZER}. Vui lòng gọi fit() trước hoặc chạy script extract embeddings.')
        instance = cls()
        instance.vec = joblib.load(PATHS.TFIDF_VECTORIZER)
        instance._fitted = True
        return instance

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('TFIDFEmbedder chưa được fit. Gọi fit(corpus) hoặc load_fitted() trước.')
        arr = self.vec.transform(texts).toarray().astype(np.float32)
        return arr

class SBERTEmbedder(BaseTextEmbedder, dim=TEXT_MODELS.DIMS['sbert']):

    def __init__(self, model_name: str=TEXT_MODELS.SBERT_NAME, device: str=DEVICE):
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers chưa được cài đặt. Vui lòng chạy: pip install sentence-transformers')
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 128

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
_CLIP_MODEL_CACHE = None

def _get_clip_model():
    global _CLIP_MODEL_CACHE
    if _CLIP_MODEL_CACHE is None:
        if open_clip is None:
            raise ImportError('open_clip chưa được cài đặt. Vui lòng chạy: pip install open-clip-torch')
        if torch is None:
            raise ImportError('torch chưa được cài đặt. Vui lòng cài đặt PyTorch trước.')
        model, _, _ = open_clip.create_model_and_transforms(CLIP.MODEL_NAME, pretrained=CLIP.PRETRAINED, device=DEVICE)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _CLIP_MODEL_CACHE = model
    return _CLIP_MODEL_CACHE

class CLIPTextEmbedder(BaseTextEmbedder, dim=TEXT_MODELS.DIMS['clip']):

    def __init__(self):
        self.model = _get_clip_model()
        self.device = DEVICE

    def _embed(self, texts: List[str]) -> np.ndarray:
        tokens = open_clip.tokenize(texts, context_length=77).to(self.device)
        with torch.no_grad():
            encoded = self.model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
        return encoded

def get_text_embedder(name: str) -> BaseTextEmbedder:
    if name == 'tfidf':
        return TFIDFEmbedder()
    elif name == 'sbert':
        return SBERTEmbedder()
    elif name == 'clip':
        return CLIPTextEmbedder()
    else:
        raise ValueError(f"Unknown text embedder: '{name}'. Available: {list(TEXT_MODELS.AVAILABLE)}")
