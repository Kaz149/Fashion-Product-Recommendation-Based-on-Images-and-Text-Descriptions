import pickle, sys, os
PROJECT_ROOT_TMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_TMP not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TMP)
import _bootstrap_env
import sys as _sys1
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from config import PATHS, CATEGORIES, EVALUATION

def main() -> None:
    PATHS.ensure_all()
    print('=' * 70)
    print('Bước 1/6: Tải metadata.csv và chuẩn hóa category')
    print('=' * 70)
    df = pd.read_csv(PATHS.DATA_PROCESSED_METADATA).reset_index(drop=True)
    N = len(df)
    df['category'] = df['category'].apply(CATEGORIES.normalize)
    print(f'✓ Đã tải {N} sản phẩm từ metadata.csv')
    print(f"✓ Category distribution:\n{df['category'].value_counts().to_string()}")
    rel_per_query: dict = {}
    query_ids: list = []
    strategy: str = ''
    print()
    print('=' * 70)
    print('Bước 2/6: Kiểm tra human_labels override (nếu có)')
    print('=' * 70)
    if PATHS.DATA_CUSTOM_HUMAN_LABELS.exists():
        print(f'✓ Phát hiện file human labels: {PATHS.DATA_CUSTOM_HUMAN_LABELS}')
        print('→ Sử dụng strategy: human_labeled')
        hl = pd.read_csv(PATHS.DATA_CUSTOM_HUMAN_LABELS)
        expected_cols = ['query_id', 'relevant_id']
        if not all((c in hl.columns for c in expected_cols)):
            raise ValueError(f'File human_labeled.csv cần có columns: {expected_cols}. Columns hiện có: {list(hl.columns)}')
        for qid, grp in hl.groupby('query_id'):
            rel_per_query[qid] = grp['relevant_id'].astype(str).tolist()
        query_ids = sorted(rel_per_query.keys())
        strategy = 'human_labeled'
    else:
        print(f'✗ Không tìm thấy file {PATHS.DATA_CUSTOM_HUMAN_LABELS}')
        print('→ Sử dụng strategy: rule-based')
        print()
        print('=' * 70)
        print('Bước 3/6: Chia tập query_set với train_test_split')
        print('=' * 70)
        print(f'EVALUATION.QUERY_SET_RATIO = {EVALUATION.QUERY_SET_RATIO}')
        print(f'EVALUATION.RANDOM_SEED = {EVALUATION.RANDOM_SEED}')
        stratify = df.category if df.category.nunique() > 1 else None
        query_set_idx, _ = train_test_split(np.arange(N), test_size=1 - EVALUATION.QUERY_SET_RATIO, random_state=EVALUATION.RANDOM_SEED, stratify=stratify)
        query_ids = df.iloc[query_set_idx]['id'].tolist()
        print(f'✓ Đã chọn {len(query_set_idx)} query (tỷ lệ {EVALUATION.QUERY_SET_RATIO * 100:.1f}%)')
        print()
        print('=' * 70)
        print('Bước 4/6: Xây dựng relevant_per_query theo rule-based')
        print('=' * 70)
        print(f'EVALUATION.GT_ATTRIBUTES = {EVALUATION.GT_ATTRIBUTES}')
        print(f'EVALUATION.GT_MATCH_MIN_COUNT = {EVALUATION.GT_MATCH_MIN_COUNT}')
        print(f'→ Điều kiện match: cùng category + match ≥ {EVALUATION.GT_MATCH_MIN_COUNT}/{len(EVALUATION.GT_ATTRIBUTES)} attribute')
        attributes = EVALUATION.GT_ATTRIBUTES
        for q_pos, query_idx in enumerate(query_set_idx):
            if (q_pos + 1) % 50 == 0 or q_pos == len(query_set_idx) - 1:
                print(f'  Đang xử lý query {q_pos + 1}/{len(query_set_idx)}...')
            qrow = df.iloc[query_idx]
            q_cat = qrow.category
            relevant_ids = []
            for i, r in df.iterrows():
                if i == query_idx:
                    continue
                if r.category != q_cat:
                    continue
                matches = 0
                for attr in attributes:
                    qv = str(qrow.get(attr, '')).strip().lower()
                    rv = str(r.get(attr, '')).strip().lower()
                    if qv and rv and (qv == rv):
                        matches += 1
                if matches >= EVALUATION.GT_MATCH_MIN_COUNT:
                    relevant_ids.append(r.id)
            rel_per_query[qrow.id] = relevant_ids
        strategy = f'rule_based_{EVALUATION.GT_MATCH_MIN_COUNT}of{len(EVALUATION.GT_ATTRIBUTES)}_plus_category'
    print()
    print('=' * 70)
    print('Bước 5/6: Lưu ground_truth.pkl')
    print('=' * 70)
    num_queries = len(query_ids)
    rel_lens = [len(v) for v in rel_per_query.values()]
    avg_relevant = float(np.mean(rel_lens)) if rel_lens else 0.0
    gt_data = {'query_ids': query_ids, 'relevant_per_query': rel_per_query, 'strategy': strategy, 'match_min_count': EVALUATION.GT_MATCH_MIN_COUNT, 'gt_attributes': list(EVALUATION.GT_ATTRIBUTES), 'num_products_total': N, 'num_queries': num_queries, 'avg_relevant_per_query': avg_relevant, 'created_at': datetime.utcnow().isoformat()}
    with open(PATHS.DATA_PROCESSED_GROUND_TRUTH, 'wb') as f:
        pickle.dump(gt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✓ Đã lưu ground truth vào: {PATHS.DATA_PROCESSED_GROUND_TRUTH}')
    print()
    print('=' * 70)
    print('Bước 6/6: Summary')
    print('=' * 70)
    print(f'Strategy               : {strategy}')
    print(f'Tổng số sản phẩm       : {N}')
    print(f'Số lượng query         : {num_queries}')
    print(f'Trung bình relevant/q  : {avg_relevant:.2f}')
    if rel_lens:
        zero_count = sum((1 for l in rel_lens if l == 0))
        if zero_count > 0:
            pct = zero_count / num_queries * 100
            print()
            print(f'⚠️  CẢNH BÁO: {zero_count}/{num_queries} query ({pct:.1f}%) KHÔNG CÓ relevant nào!')
            if EVALUATION.GT_MATCH_MIN_COUNT > 1:
                print(f'→ Gợi ý: Giảm EVALUATION.GT_MATCH_MIN_COUNT từ {EVALUATION.GT_MATCH_MIN_COUNT} xuống 1 trong config.py')
                print('  rồi chạy lại script này để có kết quả đánh giá đáng tin cậy hơn.')
        one_less_count = sum((1 for l in rel_lens if l < 2))
        if one_less_count > num_queries * 0.3 and EVALUATION.GT_MATCH_MIN_COUNT > 1:
            pct = one_less_count / num_queries * 100
            print()
            print(f'⚠️  CẢNH BÁO: {one_less_count}/{num_queries} query ({pct:.1f}%) có < 2 relevant.')
            print(f'→ Gợi ý: Có thể xem xét giảm EVALUATION.GT_MATCH_MIN_COUNT xuống 1.')
    print()
    print('✓ HOÀN THÀNH: Build Ground Truth xong!')
if __name__ == '__main__':
    main()
