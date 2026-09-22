from __future__ import annotations
import json, os, pickle, sys
from datetime import datetime, timezone
from pathlib import Path
SCRIPT_DIR_TMP = Path(__file__).resolve().parent
PROJECT_ROOT_TMP = SCRIPT_DIR_TMP.parent
if str(PROJECT_ROOT_TMP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_TMP))
import _bootstrap_env
import numpy as np
from config import PATHS
from src.dataset_adaptor import compute_dataset_fingerprint

def _get_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
    except ImportError:
        return 'unknown'
    try:
        return str(mod.__version__)
    except AttributeError:
        return 'unknown'

def _count_products_from_metadata() -> int:
    try:
        import pandas as pd
        df = pd.read_csv(PATHS.DATA_PROCESSED_METADATA, dtype=str).fillna('')
        return len(df)
    except Exception:
        return 0

def _count_products_from_index_mapping() -> int | None:
    idx_path = PATHS.EMB_INDEX_MAPPING
    if not idx_path.exists():
        return None
    try:
        with open(idx_path, 'rb') as f:
            data = pickle.load(f)
        val = data.get('n_rows')
        if val is not None:
            return int(val)
    except Exception:
        pass
    return None

def main() -> None:
    PATHS.ensure_all()
    print('=' * 72)
    print('[TASK 5] BUILD EMBEDDING MANIFEST JSON')
    print('=' * 72)
    print()
    print('[1/5] Compute dataset fingerprint ...')
    fingerprint = compute_dataset_fingerprint(PATHS.DATA_PROCESSED_METADATA, PATHS.DATA_PROCESSED_IMAGES)
    print(f'  fingerprint = {fingerprint[:16]}...{fingerprint[-8:]}')
    print('[2/5] Collect embedding info from .npy files ...')
    embedding_files = [('scratch_cnn', PATHS.EMB_SCRATCH_CNN), ('resnet50', PATHS.EMB_RESNET50), ('efficientnet_b0', PATHS.EMB_EFFNET_B0), ('clip_image', PATHS.EMB_CLIP_IMAGE), ('tfidf', PATHS.EMB_TFIDF), ('sbert', PATHS.EMB_SBERT), ('clip_text', PATHS.EMB_CLIP_TEXT)]
    embeddings_info: dict[str, dict] = {}
    for name, path in embedding_files:
        if not path.exists():
            print(f'  {name:<18} : MISSING ({path.name})')
            continue
        try:
            arr = np.load(path, mmap_mode='r')
            info = {'file': path.name, 'shape': list(arr.shape), 'dtype': str(arr.dtype), 'norm_mean': round(float(np.mean(np.linalg.norm(arr, axis=1))), 4), 'size_bytes': path.stat().st_size}
            embeddings_info[name] = info
            size_mb = info['size_bytes'] / 1000000.0
            print(f"  {name:<18} : shape={info['shape']}  norm_mean={info['norm_mean']:.4f}  {size_mb:.1f}MB")
        except Exception as e:
            print(f'  {name:<18} : ERROR reading {path.name}: {e}')
    print('[3/5] Collect library versions ...')
    libs_to_check = ['torch', 'open_clip', 'sentence_transformers', 'sklearn']
    models_version: dict[str, str] = {}
    for lib in libs_to_check:
        v = _get_version(lib)
        models_version[lib] = v
        print(f'  {lib:<22} : {v}')
    print('[4/5] Determine num_products ...')
    n_from_idx = _count_products_from_index_mapping()
    if n_from_idx is not None:
        num_products = n_from_idx
        src = 'index_mapping.pkl'
    else:
        num_products = _count_products_from_metadata()
        src = 'metadata.csv'
    print(f'  num_products = {num_products} (from {src})')
    print('[5/5] Write manifest JSON ...')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_fingerprint': fingerprint, 'num_products': num_products, 'embeddings': embeddings_info, 'models_version': models_version}
    PATHS.EMB_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS.EMB_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    sz = os.path.getsize(PATHS.EMB_MANIFEST)
    print(f'  Đã ghi {PATHS.EMB_MANIFEST.name} ({sz} bytes)')
    print()
    print(f'Manifest updated. fingerprint = {fingerprint[:16]}...')
    print('=' * 72)
if __name__ == '__main__':
    main()
