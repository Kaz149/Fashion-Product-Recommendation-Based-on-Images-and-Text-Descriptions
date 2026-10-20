from __future__ import annotations
import io
import os
import sys
for _env in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

def _patch_stdio_utf8():
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        try:
            encoding = (getattr(stream, 'encoding', '') or 'ascii').lower()
            if 'utf' not in encoding:
                buffer = getattr(stream, 'buffer', None)
                if buffer is not None:
                    new_stream = io.TextIOWrapper(buffer, encoding='utf-8', errors='replace', line_buffering=True)
                    setattr(sys, stream_name, new_stream)
        except Exception:
            pass
_patch_stdio_utf8()
del _patch_stdio_utf8
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Literal
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

@dataclass
class PATHS:
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_RAW_IMAGES: Path = PROJECT_ROOT / 'data' / 'raw' / 'images'
    DATA_CUSTOM_DIR: Path = PROJECT_ROOT / 'data' / 'custom'
    DATA_CUSTOM_METADATA: Path = PROJECT_ROOT / 'data' / 'custom' / 'metadata.csv'
    DATA_CUSTOM_HUMAN_LABELS: Path = PROJECT_ROOT / 'data' / 'custom' / 'human_labeled.csv'
    DATA_CUSTOM_FILENAME_MAP: Path = PROJECT_ROOT / 'data' / 'custom' / 'filename_to_category.json'
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / 'data' / 'processed'
    DATA_PROCESSED_IMAGES: Path = PROJECT_ROOT / 'data' / 'processed' / 'images'
    DATA_PROCESSED_METADATA: Path = PROJECT_ROOT / 'data' / 'processed' / 'metadata.csv'
    DATA_PROCESSED_GROUND_TRUTH: Path = PROJECT_ROOT / 'data' / 'processed' / 'ground_truth.pkl'
    DATA_EMBEDDINGS_DIR: Path = PROJECT_ROOT / 'data' / 'embeddings'
    EMB_RESNET50: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'resnet50_embeddings.npy'
    EMB_EFFNET_B0: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'efficientnet_b0_embeddings.npy'
    EMB_CLIP_IMAGE: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'clip_image_embeddings.npy'
    EMB_TFIDF: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'tfidf_embeddings.npy'
    EMB_SBERT: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'sbert_embeddings.npy'
    EMB_CLIP_TEXT: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'clip_text_embeddings.npy'
    EMB_INDEX_MAPPING: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'index_mapping.pkl'
    EMB_MANIFEST: Path = PROJECT_ROOT / 'data' / 'embeddings' / 'embedding_manifest.json'
    DATA_RESULTS_DIR: Path = PROJECT_ROOT / 'data' / 'results'
    EVAL_RESULTS_CSV: Path = PROJECT_ROOT / 'data' / 'results' / 'evaluation_results.csv'
    MODELS_DIR: Path = PROJECT_ROOT / 'models'
    TFIDF_VECTORIZER: Path = PROJECT_ROOT / 'models' / 'tfidf_vectorizer.pkl'
    REPORTS_DIR: Path = PROJECT_ROOT / 'reports'
    REPORTS_FIGURES_DIR: Path = PROJECT_ROOT / 'reports' / 'figures'
    ERROR_ANALYSIS_MD: Path = PROJECT_ROOT / 'reports' / 'error_analysis.md'
    TESTS_DIR: Path = PROJECT_ROOT / 'tests'
    TESTS_FIXTURE_IMAGES: Path = PROJECT_ROOT / 'tests' / 'fixture_images'

    def ensure_all(self) -> None:
        for p in [self.DATA_RAW_IMAGES, self.DATA_CUSTOM_DIR, self.DATA_PROCESSED_DIR, self.DATA_PROCESSED_IMAGES, self.DATA_EMBEDDINGS_DIR, self.DATA_RESULTS_DIR, self.MODELS_DIR, self.REPORTS_DIR, self.REPORTS_FIGURES_DIR, self.TESTS_DIR, self.TESTS_FIXTURE_IMAGES]:
            p.mkdir(parents=True, exist_ok=True)

@dataclass
class IMAGE_MODELS:
    AVAILABLE: Tuple[str, ...] = ('resnet', 'effnet', 'clip')
    DEFAULT: str = 'resnet'
    DIMS: Dict[str, int] = field(default_factory=lambda: {'resnet': 2048, 'effnet': 1280, 'clip': 512})

@dataclass
class TEXT_MODELS:
    AVAILABLE: Tuple[str, ...] = ('tfidf', 'sbert', 'clip')
    DEFAULT: str = 'sbert'
    DIMS: Dict[str, int] = field(default_factory=lambda: {'tfidf': 5000, 'sbert': 384, 'clip': 512})
    SBERT_NAME: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    TFIDF_MAX_FEATURES: int = 5000
    TFIDF_NGRAM: Tuple[int, int] = (1, 2)
    TFIDF_MIN_DF: int = 1
    STOP_WORDS = None

@dataclass
class CLIP:
    MODEL_NAME: str = 'ViT-B-32'
    PRETRAINED: str = 'laion2b_s34b_b79k'
    EMBED_DIM: int = 512

@dataclass
class TRAINING:
    RANDOM_SEED: int = 42
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    EPOCHS: int = 30
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0001
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    EARLY_STOPPING_PATIENCE: int = 5
    LR_SCHED_PATIENCE: int = 3
    LR_SCHED_FACTOR: float = 0.5
    DROPOUT: float = 0.3
    EMBED_DIM: int = 512
    AUG_COLOR_JITTER: float = 0.15
    NUM_WORKERS: int = 0

@dataclass
class RETRIEVAL:
    DEFAULT_TOP_K: int = 10
    DEFAULT_MAX_K: int = 20
    DEFAULT_IMAGE: Literal['resnet', 'effnet', 'clip'] = 'resnet'
    DEFAULT_TEXT: Literal['tfidf', 'sbert', 'clip'] = 'sbert'
    METRIC: Literal['cosine'] = 'cosine'
    ALGORITHM: Literal['brute', 'ball_tree', 'kd_tree', 'auto'] = 'brute'
    FUSION_ALPHA_DEFAULT: float = 0.5
    FUSION_ALPHA_GRID: Tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    FUSION_BEST_ALPHA: float = 0.5
    OUTFIT_MATCH_FILTER_CATEGORY: bool = True

@dataclass
class CATEGORIES:
    FASHION: Tuple[str, ...] = ('shirt', 't-shirt', 'tshirt', 'tee', 'top', 'blouse', 'polo', 'sweater', 'hoodie', 'jacket', 'coat', 'trouser', 'pant', 'pants', 'jeans', 'jean', 'short', 'shorts', 'skirt', 'shoes', 'shoe', 'sneaker', 'sneakers', 'boot', 'boots', 'sandal', 'sandals', 'footwear', 'loafer', 'bag', 'purse', 'handbag', 'backpack', 'tote', 'clutch', 'wallet', 'watch', 'wristwatch', 'dress', 'gown', 'underwear', 'underpant', 'brief', 'bra', 'panties', 'lingerie')
    ALIAS: Dict[str, str] = field(default_factory=lambda: {'t-shirt': 'shirt', 'tshirt': 'shirt', 'tee': 'shirt', 'top': 'shirt', 'blouse': 'shirt', 'polo': 'shirt', 'sweater': 'shirt', 'hoodie': 'shirt', 'jacket': 'shirt', 'coat': 'shirt', 'pant': 'trouser', 'pants': 'trouser', 'jeans': 'trouser', 'jean': 'trouser', 'short': 'trouser', 'shorts': 'trouser', 'skirt': 'dress', 'shoe': 'shoes', 'sneaker': 'shoes', 'sneakers': 'shoes', 'boot': 'shoes', 'boots': 'shoes', 'sandal': 'shoes', 'sandals': 'shoes', 'footwear': 'shoes', 'loafer': 'shoes', 'purse': 'bag', 'handbag': 'bag', 'backpack': 'bag', 'tote': 'bag', 'clutch': 'bag', 'wallet': 'bag', 'wristwatch': 'watch', 'gown': 'dress', 'underpant': 'underwear', 'brief': 'underwear', 'bra': 'underwear', 'panties': 'underwear', 'lingerie': 'underwear'})
    VALID_FASHION: Tuple[str, ...] = ('shirt', 'trouser', 'shoes', 'bag', 'watch', 'dress', 'underwear')
    OUTFIT_MAP: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {'shirt': ('trouser', 'shoes'), 'trouser': ('shirt', 'shoes'), 'shoes': ('trouser', 'shirt'), 'dress': ('shoes', 'bag'), 'bag': ('dress', 'shirt', 'trouser')})
    COLORS: Tuple[str, ...] = ('white', 'black', 'blue', 'navy', 'gray', 'pink', 'red', 'olive', 'brown', 'beige', 'green', 'yellow', 'purple', 'orange', 'cream', 'khaki', 'maroon', 'teal', 'cyan', 'mustard', 'coral', 'lavender', 'silver', 'charcoal', 'gold', 'mint')
    MATERIALS: Tuple[str, ...] = ('cotton', 'polyester', 'leather', 'denim', 'linen', 'canvas', 'silk', 'wool', 'suede', 'nylon', 'rubber', 'satin')
    STYLES: Tuple[str, ...] = ('casual', 'formal', 'sporty', 'office', 'vintage', 'minimalist', 'streetwear', 'elegant', 'classic', 'modern')

    def normalize(self, cat: str) -> str:
        if not cat:
            return 'other'
        c = cat.strip().lower()
        if c in self.ALIAS:
            return self.ALIAS[c]
        if c in self.FASHION:
            return c
        return 'other'

@dataclass
class EVALUATION:
    RANDOM_SEED: int = 42
    QUERY_SET_RATIO: float = 0.15
    GT_MATCH_MIN_COUNT: int = 2
    GT_ATTRIBUTES: Tuple[str, ...] = ('color', 'material', 'style')
    K_LIST: Tuple[int, ...] = (1, 5, 10, 20)
    MAIN_K: int = 10
    REPORT_K: Tuple[int, int, int, int, int] = (5, 10, 5, 10, 5)

@dataclass
class ERROR_ANALYSIS:
    NUM_CASES: int = 25
    TYPES: Tuple[str, ...] = ('IMAGE_EMBEDDING_FAILURE', 'TEXT_METADATA_FAILURE', 'FUSION_ALPHA', 'GROUND_TRUTH_FLAW')
    ALPHA_TUNE_VALUES: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'
DEVICE: str = os.environ.get('REC_DEVICE', _get_device())
PATHS = PATHS()
IMAGE_MODELS = IMAGE_MODELS()
TEXT_MODELS = TEXT_MODELS()
CLIP = CLIP()
TRAINING = TRAINING()
RETRIEVAL = RETRIEVAL()
CATEGORIES = CATEGORIES()
EVALUATION = EVALUATION()
ERROR_ANALYSIS = ERROR_ANALYSIS()
__all__ = ['PROJECT_ROOT', 'DEVICE', 'PATHS', 'IMAGE_MODELS', 'TEXT_MODELS', 'CLIP', 'TRAINING', 'RETRIEVAL', 'CATEGORIES', 'EVALUATION', 'ERROR_ANALYSIS']
