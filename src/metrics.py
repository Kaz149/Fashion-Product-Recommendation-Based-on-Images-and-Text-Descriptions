from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from config import EVALUATION

def _as_set(rel: Any) -> set:
    if isinstance(rel, set):
        return {str(x) for x in rel}
    if isinstance(rel, (list, tuple)):
        return {str(x) for x in rel}
    if isinstance(rel, np.ndarray):
        return {str(x) for x in rel.tolist()}
    if isinstance(rel, pd.Series):
        return {str(x) for x in rel.tolist()}
    return {str(rel)}

def precision_at_k(retrieved_ids: list, relevant_ids: Any, k: int) -> float:
    if k <= 0:
        return 0.0
    rel_set = _as_set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum((1 for rid in top_k if str(rid) in rel_set))
    return hits / k

def recall_at_k(retrieved_ids: list, relevant_ids: Any, k: int) -> float:
    rel_set = _as_set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum((1 for rid in top_k if str(rid) in rel_set))
    return hits / len(rel_set)

def average_precision_at_k(retrieved_ids: list, relevant_ids: Any, k: int) -> float:
    rel_set = _as_set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    score = 0.0
    num_hits = 0
    n_retrieved = min(k, len(retrieved_ids))
    for i in range(n_retrieved):
        rid = str(retrieved_ids[i])
        if rid in rel_set:
            num_hits += 1
            p_at_i = num_hits / (i + 1)
            score += p_at_i
    denom = min(k, len(rel_set))
    return score / denom if denom > 0 else 0.0

def mean_average_precision_at_k(all_retrieved: Dict[Any, list], all_relevant: Dict[Any, list], k: int) -> float:
    all_keys = set(all_retrieved.keys()) | set(all_relevant.keys())
    aps = []
    for key in all_keys:
        rel = all_relevant.get(key, [])
        if len(_as_set(rel)) == 0:
            continue
        retr = all_retrieved.get(key, [])
        aps.append(average_precision_at_k(retr, rel, k))
    return float(np.mean(aps)) if aps else 0.0

def f1_at_k(retrieved_ids: list, relevant_ids: Any, k: int) -> float:
    eps = 1e-12
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * p * r / (p + r + eps)

def evaluate_method(method_name: str, retriever: Any, query_ids: list, relevant_per_query: Dict, mode_config: dict, k_list: Optional[Tuple[int, ...]]=None) -> Dict:
    if k_list is None:
        k_list = EVALUATION.K_LIST
    max_k = max(k_list)
    metadata_df = retriever.load_metadata()
    per_query_p = {k: [] for k in k_list}
    per_query_r = {k: [] for k in k_list}
    per_query_ap = {k: [] for k in k_list}
    per_query_f1 = {k: [] for k in k_list}
    n_queries = len(query_ids)
    valid_query_count = 0
    mode = mode_config['mode']
    for qid in query_ids:
        qid_str = str(qid)
        relevant_ids = relevant_per_query.get(qid_str, [])
        rel_set = _as_set(relevant_ids)
        if len(rel_set) == 0:
            continue
        valid_query_count += 1
        row_mask = metadata_df['id'].astype(str) == qid_str
        if not row_mask.any():
            try:
                row_mask = metadata_df['id'] == int(qid)
            except Exception:
                pass
        if not row_mask.any():
            continue
        row = metadata_df[row_mask].iloc[0]
        img_path = str(row.get('image_path', ''))
        product_name = str(row.get('product_name', '')).strip()
        description = str(row.get('description', '')).strip()
        category = str(row.get('category', '')).strip()
        color = str(row.get('color', '')).strip()
        material = str(row.get('material', '')).strip()
        style = str(row.get('style', '')).strip()
        query_text = ' '.join([product_name, description, category, color, material, style]).strip()
        results = []
        if mode == 'image':
            results = retriever.search_by_image(img_path, model=mode_config['model'], top_k=max_k + 1)
        elif mode == 'text':
            if not query_text:
                query_text = product_name or 'fashion item'
            results = retriever.search_by_text(query_text, model=mode_config['model'], top_k=max_k + 1)
            results = [r for r in results if str(r.get('id', '')) != qid_str]
        elif mode == 'multimodal':
            if not query_text:
                query_text = product_name or 'fashion item'
            alpha = mode_config.get('alpha', 0.5)
            results = retriever.search_multimodal(img_path=img_path, text=query_text, alpha=alpha, top_k=max_k + 1)
            results = [r for r in results if str(r.get('id', '')) != qid_str]
        results = results[:max_k]
        retrieved_ids = [str(r.get('id', '')) for r in results]
        for k in k_list:
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            ap = average_precision_at_k(retrieved_ids, relevant_ids, k)
            f1 = f1_at_k(retrieved_ids, relevant_ids, k)
            per_query_p[k].append(p)
            per_query_r[k].append(r)
            per_query_ap[k].append(ap)
            per_query_f1[k].append(f1)
    result: Dict[str, Any] = {'method': method_name, 'n_queries': n_queries, 'n_valid_queries': valid_query_count}
    for k in k_list:
        vals = per_query_r[k]
        result[f'Recall@{k}'] = float(np.mean(vals)) if vals else 0.0
    for k in k_list:
        vals = per_query_p[k]
        result[f'Precision@{k}'] = float(np.mean(vals)) if vals else 0.0
    main_ap_vals = per_query_ap.get(EVALUATION.MAIN_K, [])
    result[f'mAP@{EVALUATION.MAIN_K}'] = float(np.mean(main_ap_vals)) if main_ap_vals else 0.0
    for k in k_list:
        vals = per_query_f1[k]
        result[f'F1@{k}'] = float(np.mean(vals)) if vals else 0.0
    return result
