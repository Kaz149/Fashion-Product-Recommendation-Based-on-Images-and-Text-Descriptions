"""Build only TF-IDF from the current catalog; keep the other encoders intact."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import _bootstrap_env
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PATHS, TEXT_MODELS
from src.dataset_adaptor import MetadataGenerator, compute_dataset_fingerprint


def main() -> None:
    df = pd.read_csv(PATHS.DATA_PROCESSED_METADATA, dtype=str).fillna('')
    fields = ['id', 'product_name', 'description', 'color', 'material', 'style']
    missing = [column for column in fields if column not in df.columns]
    if missing:
        raise ValueError(f'Metadata thiếu cột: {missing}')
    ids = df['id'].tolist()
    if not ids or df['id'].str.strip().eq('').any() or df['id'].duplicated().any():
        raise ValueError('Danh mục phải có sản phẩm và ID không được trống hoặc trùng.')
    with PATHS.EMB_INDEX_MAPPING.open('rb') as handle:
        mapping = pickle.load(handle)
    if ids != [str(value) for value in mapping['product_ids']]:
        raise ValueError('Thứ tự ID trong metadata và index_mapping không khớp; chưa ghi thay đổi.')
    manifest = json.loads(PATHS.EMB_MANIFEST.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('embeddings'), dict):
        raise ValueError('Manifest không hợp lệ; chưa ghi thay đổi.')

    # Read existing descriptions without generating or guessing new attributes.
    text_fields = fields[1:]
    empty = df[text_fields].apply(lambda column: column.str.strip()).eq('').all(axis=1)
    if empty.any():
        raise ValueError(f'{int(empty.sum())} sản phẩm không có văn bản; chưa ghi thay đổi.')
    corpus = MetadataGenerator.build_corpus_list(df.to_dict(orient='records'))
    vectorizer = TfidfVectorizer(
        max_features=TEXT_MODELS.TFIDF_MAX_FEATURES,
        ngram_range=TEXT_MODELS.TFIDF_NGRAM,
        min_df=TEXT_MODELS.TFIDF_MIN_DF,
        stop_words=TEXT_MODELS.STOP_WORDS,
    )
    sparse_vectors = vectorizer.fit_transform(corpus).astype(np.float32)
    if not np.isfinite(sparse_vectors.data).all():
        raise ValueError('TF-IDF chứa giá trị không hữu hạn; chưa ghi thay đổi.')
    norms = np.sqrt(np.asarray(sparse_vectors.multiply(sparse_vectors).sum(axis=1)).ravel())
    if np.any(norms == 0):
        raise ValueError('Có sản phẩm không có từ hợp lệ; chưa ghi thay đổi.')
    # Existing retrieval and evaluation scripts load this .npy format.
    # Keep that format for compatibility; fitting itself uses a sparse matrix.
    timestamp = datetime.now(timezone.utc)
    backup_dir = PATHS.PROJECT_ROOT / 'data' / 'backups' / ('tfidf_' + timestamp.strftime('%Y%m%d_%H%M%S_%f'))
    targets = [PATHS.TFIDF_VECTORIZER, PATHS.EMB_TFIDF, PATHS.EMB_MANIFEST]

    with tempfile.TemporaryDirectory(prefix='tfidf_build_', dir=PATHS.DATA_EMBEDDINGS_DIR) as tmp:
        stage = Path(tmp)
        vectorizer_file = stage / PATHS.TFIDF_VECTORIZER.name
        vectors_file = stage / PATHS.EMB_TFIDF.name
        manifest_file = stage / PATHS.EMB_MANIFEST.name
        joblib.dump(vectorizer, vectorizer_file)
        saved = np.lib.format.open_memmap(vectors_file, mode='w+', dtype=np.float32, shape=sparse_vectors.shape)
        for start in range(0, len(ids), 128):
            saved[start:start + 128] = sparse_vectors[start:start + 128].toarray()
        saved.flush()
        del saved  # Close the mapping before replacing the file on Windows.
        manifest['embeddings']['tfidf'] = {
            'file': PATHS.EMB_TFIDF.name,
            'shape': list(sparse_vectors.shape),
            'dtype': 'float32',
            'norm_mean': round(float(norms.mean()), 4),
            'size_bytes': vectors_file.stat().st_size,
            'created_at': timestamp.isoformat(),
            'sklearn_version': sklearn.__version__,
            'dataset_fingerprint': compute_dataset_fingerprint(PATHS.DATA_PROCESSED_METADATA, PATHS.DATA_PROCESSED_IMAGES),
            'metadata_sha256': hashlib.sha256(PATHS.DATA_PROCESSED_METADATA.read_bytes()).hexdigest(),
            'product_ids_sha256': hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode('utf-8')).hexdigest(),
            'vectorizer_sha256': hashlib.sha256(vectorizer_file.read_bytes()).hexdigest(),
            'vectors_sha256': hashlib.sha256(vectors_file.read_bytes()).hexdigest(),
            'text_builder': 'MetadataGenerator.build_corpus_text',
            'parameters': {
                'max_features': TEXT_MODELS.TFIDF_MAX_FEATURES,
                'ngram_range': list(TEXT_MODELS.TFIDF_NGRAM),
                'min_df': TEXT_MODELS.TFIDF_MIN_DF,
                'stop_words': TEXT_MODELS.STOP_WORDS,
            },
        }
        # Preserve the global fingerprint and all other model entries:
        # rebuilding TF-IDF cannot certify that other embeddings are current.
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        backups = {}
        for target in targets:
            if target.exists():
                backup = backup_dir / target.relative_to(PATHS.PROJECT_ROOT)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                backups[target] = backup
        installed = []
        try:
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / target.name, target)
                installed.append(target)
        except Exception:
            for target in reversed(installed):
                if target in backups:
                    shutil.copy2(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise

    print(f'TF-IDF rebuilt: {len(ids)} products, {len(vectorizer.vocabulary_)} vocabulary entries.')
    print(f'Embedding shape: {tuple(sparse_vectors.shape)}')
    print(f'Backup: {backup_dir}')
    print('Restart Streamlit, choose tfidf, and try: red shirt')
    print('Existing evaluation reports must be rerun before claiming results for this new TF-IDF.')


if __name__ == '__main__':
    main()
