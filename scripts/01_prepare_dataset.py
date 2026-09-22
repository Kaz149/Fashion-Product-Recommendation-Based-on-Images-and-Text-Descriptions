from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
import _bootstrap_env
from tqdm import tqdm
from config import PATHS
from src.dataset_adaptor import DatasetLoader, MetadataGenerator, sanitize_images, save_metadata, filter_fashion_only
BANNER = '=' * 70
STEP_FMT = '\n{BANNER}\n  [STEP {n}/5] {title}\n{BANNER}'

def _banner_step(n: int, title: str) -> None:
    print(STEP_FMT.format(BANNER=BANNER, n=n, title=title))

def _final_summary(rows: list[dict]) -> None:
    n = len(rows)
    cats = Counter((r.get('category', 'other') for r in rows))
    k = len(cats)
    top5 = cats.most_common(5)
    top_str = ', '.join((f'{c}: {cnt}' for c, cnt in top5))
    print()
    print(BANNER)
    print('  >>> Dataset preparation COMPLETE <<<')
    print(BANNER)
    print(f'  Dataset prepared: {n} products, {k} categories')
    print(f'  Top categories: {top_str}')
    print(f'  Saved to: {PATHS.DATA_PROCESSED_METADATA}')
    print(f'  Images dir: {PATHS.DATA_PROCESSED_IMAGES}')
    print(BANNER)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Prepare fashion dataset: load -> fill metadata -> sanitize images -> save metadata.csv')
    ap.add_argument('--dataset-format', '-f', choices=['A', 'B', 'C'], default='C', help='Dataset format: A=folder-per-category, B=CSV+images, C=flat folder (auto-predict category)')
    ap.add_argument('--dataset-path', '-p', type=str, default=str(PATHS.DATA_RAW_IMAGES), help=f'Path to raw dataset (default: {PATHS.DATA_RAW_IMAGES})')
    ap.add_argument('--seed', '-s', type=int, default=42, help='Random seed for deterministic metadata generation (default: 42)')
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    fmt: str = args.dataset_format
    dataset_path = Path(args.dataset_path)
    seed: int = args.seed
    print()
    print(BANNER)
    print('  MULTIMODAL FASHION RECOMMENDATION SYSTEM')
    print('  STEP 01: PREPARE DATASET')
    print(BANNER)
    print(f'  Format       : {fmt}')
    print(f'  Dataset path : {dataset_path}')
    print(f'  Seed         : {seed}')
    print(f'  Output meta  : {PATHS.DATA_PROCESSED_METADATA}')
    print(f'  Output imgs  : {PATHS.DATA_PROCESSED_IMAGES}')
    print(BANNER)
    _banner_step(1, 'Ensure directories exist')
    PATHS.ensure_all()
    print('  [OK] All directories created/existed.')
    _banner_step(2, f'Load dataset (format {fmt})')
    loader = DatasetLoader(format=fmt, dataset_path=dataset_path)
    print('  Loading rows...')
    rows = loader.load_rows()
    print(f'  [OK] Loaded {len(rows)} rows from format {fmt}.')
    if not rows:
        print('  [WARN] Empty dataset. Nothing to process. Exit.')
        return 0
    _banner_step(3, 'Fill metadata defaults + FASHION ONLY FILTER')
    gen = MetadataGenerator(seed=seed)
    new_rows: list[dict] = []
    for r in tqdm(rows, desc='  Filling metadata', unit='row', ncols=80):
        new_rows.append(gen.fill_defaults(r))
    rows = new_rows
    n_before = len(rows)
    rows = filter_fashion_only(rows)
    n_after = len(rows)
    n_dropped = n_before - n_after
    print(f'  [OK] Metadata generated for {n_before} products.')
    if n_dropped > 0:
        print(f'  [FASHION-ONLY FILTER] Dropped {n_dropped} non-fashion products (bottle/ball/other/non-fashion categories). Remaining: {n_after} products.')
    else:
        print(f'  [FASHION-ONLY FILTER] OK: All {n_after} products belong to VALID FASHION categories (shirt/trouser/shoes/bag/watch/dress/underwear).')
    if not rows:
        print('  [ERROR] 0 products remaining after fashion-only filter. Check your dataset category names.')
        return 1
    _banner_step(4, 'Sanitize + copy images (resize 224x224)')
    raw_pairs: list[tuple] = []
    for r in rows:
        raw_path = r.get('_raw_image_path') or r.get('image_path')
        if raw_path:
            raw_pairs.append((r['id'], Path(raw_path)))
    print(f'  Processing {len(raw_pairs)} images...')
    success, failed = sanitize_images(raw_paths_with_ids=raw_pairs, dst_dir=PATHS.DATA_PROCESSED_IMAGES, size=224)
    id_to_new_path = {item_id: new_path for item_id, new_path in success}
    failed_set = set(failed)
    keep_rows: list[dict] = []
    for r in tqdm(rows, desc='  Updating image paths', unit='row', ncols=80):
        rid = r['id']
        if rid in failed_set:
            continue
        if rid in id_to_new_path:
            r['image_path'] = str(id_to_new_path[rid])
        for k in list(r.keys()):
            if k.startswith('_'):
                r.pop(k, None)
        keep_rows.append(r)
    rows = keep_rows
    print(f'  [OK] Images processed: success={len(success)}, failed={len(failed)}')
    if failed:
        print(f'  [WARN] Failed image IDs (first 10): {list(failed)[:10]}')
    if not rows:
        print('  [ERROR] 0 valid products after image sanitization. Exit.')
        return 1
    _banner_step(5, f'Save metadata.csv -> {PATHS.DATA_PROCESSED_METADATA.name}')
    save_metadata(rows, PATHS.DATA_PROCESSED_METADATA)
    print(f'  [OK] Saved {len(rows)} rows to {PATHS.DATA_PROCESSED_METADATA}')
    _final_summary(rows)
    return 0
if __name__ == '__main__':
    sys.exit(main())
