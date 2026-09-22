from __future__ import annotations
import argparse, os, pickle, sys, warnings
from pathlib import Path
SCRIPT_DIR_TMP = Path(__file__).resolve().parent
PROJECT_ROOT_TMP = SCRIPT_DIR_TMP.parent
if str(PROJECT_ROOT_TMP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_TMP))
import _bootstrap_env
import numpy as np
import pandas as pd
from config import PATHS, IMAGE_MODELS, TEXT_MODELS, DEVICE
from src.dataset_adaptor import MetadataGenerator, is_embedding_stale
from src.image_embedder import get_image_embedder
from src.text_embedder import get_text_embedder, TFIDFEmbedder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trích xuất tất cả embeddings (ảnh + text) cho dataset')
    parser.add_argument('--type', type=str, choices=['image', 'text', 'all'], default='all', help='Loại embedding cần trích xuất (default: all)')
    parser.add_argument('--skip-existing', action='store_true', default=False, help='Bỏ qua nếu file npy đã tồn tại và không stale (default: False)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size cho embedding (default: 32)')
    return parser.parse_args()

def _safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ''
    return str(val).strip()

def main() -> None:
    args = parse_args()
    PATHS.ensure_all()
    result_entries: list[dict] = []
    print('=' * 72)
    print('[TASK 5] EXTRACT ALL EMBEDDINGS + INDEX MAPPING')
    print('=' * 72)
    print(f'Device           = {DEVICE}')
    print(f'Type             = {args.type}')
    print(f'Skip existing    = {args.skip_existing}')
    print(f'Batch size       = {args.batch_size}')
    print()
    print('-' * 72)
    print('[STEP 0] Load metadata & build corpus')
    print('-' * 72)
    meta_path = PATHS.DATA_PROCESSED_METADATA
    if not meta_path.exists():
        raise FileNotFoundError(f'Không tìm thấy metadata.csv tại {meta_path}. Vui lòng chạy scripts/01_prepare_dataset.py trước.')
    df = pd.read_csv(meta_path, dtype=str).fillna('')
    N = len(df)
    print(f'Loaded metadata: {N} rows, columns={list(df.columns)}')
    image_paths = [Path(p) for p in df['image_path'].tolist()]
    product_ids = df['id'].tolist()
    try:
        rows = df.to_dict(orient='records')
        corpus = MetadataGenerator.build_corpus_list(rows)
        print(f'Built corpus via MetadataGenerator.build_corpus_list (N={len(corpus)})')
    except Exception:
        corpus = []
        for _, r in df.iterrows():
            pn = _safe_str(r.get('product_name'))
            ds = _safe_str(r.get('description'))
            cl = _safe_str(r.get('color'))
            mt = _safe_str(r.get('material'))
            st = _safe_str(r.get('style'))
            attr = ' '.join((x for x in [cl, mt, st] if x)).strip()
            corpus.append(f'{pn} [SEP] {ds} [SEP] {attr}')
        print(f'Built corpus via fallback concatenate (N={len(corpus)})')
    stale = is_embedding_stale(manifest_path=PATHS.EMB_MANIFEST, metadata_csv_path=PATHS.DATA_PROCESSED_METADATA, processed_images_dir=PATHS.DATA_PROCESSED_IMAGES)
    print(f'Embedding stale status = {stale}')
    print()
    if args.type in {'image', 'all'}:
        print('-' * 72)
        print('[STEP 1] Extract IMAGE embeddings (4 models)')
        print('-' * 72)
        image_embedders = [('scratch_cnn', 'scratch', PATHS.EMB_SCRATCH_CNN), ('resnet50', 'resnet', PATHS.EMB_RESNET50), ('efficientnet_b0', 'effnet', PATHS.EMB_EFFNET_B0), ('clip_image', 'clip', PATHS.EMB_CLIP_IMAGE)]
        for idx, (name, cls_name, path) in enumerate(image_embedders, 1):
            dim = IMAGE_MODELS.DIMS[cls_name]
            banner = f'[Image {idx}/4] {name} embedding ... dim={dim}'
            print()
            print(banner)
            skip_reason = None
            if args.skip_existing and path.exists() and (not stale):
                skip_reason = 'file exists + not stale'
            if skip_reason:
                print(f'  SKIP ({skip_reason})')
                result_entries.append({'group': 'image', 'name': name, 'path': path, 'status': 'skipped', 'reason': skip_reason})
                continue
            arr = None
            scratch_warned = False
            try:
                embedder = get_image_embedder(cls_name)
                arr = embedder.embed_batch(image_paths, batch_size=args.batch_size, verbose=True)
            except Exception as e:
                if cls_name == 'scratch':
                    warnings.warn(f'scratch checkpoint not found, skipping. Run 02_train_scratch_cnn.py first. (Lý do chi tiết: {e})', stacklevel=2)
                    scratch_warned = True
                    arr = np.zeros((N, IMAGE_MODELS.DIMS['scratch']), dtype=np.float32)
                else:
                    raise
            arr = arr.astype(np.float32)
            np.save(path, arr)
            sz_mb = os.path.getsize(path) / 1000000.0
            status = 'saved (scratch-zeros-placeholder)' if scratch_warned else 'saved'
            print(f'  Saved {path.name} shape {arr.shape} file_size_mb={sz_mb:.1f}MB')
            result_entries.append({'group': 'image', 'name': name, 'path': path, 'status': status, 'reason': '' if not scratch_warned else 'scratch ckpt missing → zeros placeholder'})
    if args.type in {'text', 'all'}:
        print()
        print('-' * 72)
        print('[STEP 2] Extract TEXT embeddings (3 models)')
        print('-' * 72)
        text_embedders = [('tfidf', 'tfidf', PATHS.EMB_TFIDF), ('sbert', 'sbert', PATHS.EMB_SBERT), ('clip_text', 'clip', PATHS.EMB_CLIP_TEXT)]
        for idx, (name, cls_name, path) in enumerate(text_embedders, 1):
            dim = TEXT_MODELS.DIMS[cls_name]
            banner = f'[Text {idx}/3] {name} ... dim={dim}'
            print()
            print(banner)
            skip_reason = None
            if args.skip_existing and path.exists() and (not stale):
                skip_reason = 'file exists + not stale'
            if skip_reason:
                print(f'  SKIP ({skip_reason})')
                result_entries.append({'group': 'text', 'name': name, 'path': path, 'status': 'skipped', 'reason': skip_reason})
                continue
            if cls_name == 'tfidf':
                if PATHS.TFIDF_VECTORIZER.exists():
                    print(f'  Loading fitted TF-IDF vectorizer from {PATHS.TFIDF_VECTORIZER.name}')
                    tfidf = TFIDFEmbedder.load_fitted()
                else:
                    print(f'  Fitting new TF-IDF vectorizer on corpus (len={len(corpus)})')
                    tfidf = TFIDFEmbedder()
                    tfidf.fit(corpus)
                arr = tfidf.embed_batch(corpus, batch_size=args.batch_size, verbose=True)
            else:
                embedder = get_text_embedder(cls_name)
                arr = embedder.embed_batch(corpus, batch_size=args.batch_size, verbose=True)
            arr = arr.astype(np.float32)
            np.save(path, arr)
            sz_mb = os.path.getsize(path) / 1000000.0
            print(f'  Saved {path.name} shape {arr.shape} file_size_mb={sz_mb:.1f}MB')
            result_entries.append({'group': 'text', 'name': name, 'path': path, 'status': 'saved', 'reason': ''})
    print()
    print('-' * 72)
    print('[STEP 3] Save index_mapping.pkl')
    print('-' * 72)
    index_data = {'product_ids': product_ids, 'image_paths': [str(p) for p in image_paths], 'n_rows': N}
    with open(PATHS.EMB_INDEX_MAPPING, 'wb') as f:
        pickle.dump(index_data, f)
    print(f'Saved index_mapping.pkl, N = {N}')
    print()
    print('=' * 72)
    print('[SUMMARY] Embedding extraction results')
    print('=' * 72)
    header = f"{'Group':<8} {'Name':<18} {'Status':<30} {'Size (MB)':<10}"
    print(header)
    print('-' * len(header))
    for e in result_entries:
        p: Path = e['path']
        if p.exists():
            sz = f'{os.path.getsize(p) / 1000000.0:.1f}'
        else:
            sz = 'MISSING'
        status_str = e['status']
        if e['reason']:
            status_str += f" [{e['reason']}]"
        print(f"{e['group']:<8} {e['name']:<18} {status_str:<30} {sz:<10}")
    print()
    print(f'Tổng số file xử lý: {len(result_entries)}')
    saved_count = sum((1 for e in result_entries if 'saved' in e['status']))
    skip_count = sum((1 for e in result_entries if e['status'] == 'skipped'))
    print(f'  Đã tạo / ghi đè : {saved_count}')
    print(f'  Đã bỏ qua        : {skip_count}')
    print()
    print('Hoàn thành. Chạy scripts/04_build_embedding_manifest.py để cập nhật manifest.')
    print('=' * 72)
if __name__ == '__main__':
    main()
