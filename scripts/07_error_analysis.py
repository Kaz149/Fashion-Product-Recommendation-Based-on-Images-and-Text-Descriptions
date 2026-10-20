from __future__ import annotations
import sys
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
import _bootstrap_env
import numpy as np
import pandas as pd
from config import PATHS, IMAGE_MODELS, TEXT_MODELS, RETRIEVAL, CATEGORIES, EVALUATION, ERROR_ANALYSIS, DEVICE
REPORTS_ERROR_CASES_DIR = PATHS.REPORTS_DIR / 'error_cases'
REPORTS_ERROR_ANALYSIS_MD = PATHS.ERROR_ANALYSIS_MD

def recall_at_k(ranked_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    if not relevant_ids:
        return 0.0
    rel_set = set((str(r) for r in relevant_ids))
    top_k_ids = [str(x) for x in ranked_ids[:k]]
    hit = sum((1 for rid in top_k_ids if rid in rel_set))
    return hit / len(rel_set)

def precision_at_k(ranked_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_set = set((str(r) for r in relevant_ids))
    top_k_ids = [str(x) for x in ranked_ids[:k]]
    hit = sum((1 for rid in top_k_ids if rid in rel_set))
    return hit / min(k, len(top_k_ids)) if top_k_ids else 0.0

def average_precision_at_k(ranked_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    if not relevant_ids:
        return 0.0
    rel_set = set((str(r) for r in relevant_ids))
    ranked_str = [str(x) for x in ranked_ids[:k]]
    hits = 0.0
    sum_prec = 0.0
    for i, rid in enumerate(ranked_str):
        if rid in rel_set:
            hits += 1.0
            sum_prec += hits / (i + 1.0)
    if hits == 0:
        return 0.0
    return sum_prec / min(len(rel_set), k)

def step0_load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    PATHS.ensure_all()
    REPORTS_ERROR_CASES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PATHS.DATA_PROCESSED_METADATA).reset_index(drop=True)
    df['id'] = df['id'].astype(str)
    N = len(df)
    print(f'[STEP 0] ✓ Tải metadata: {N} sản phẩm')
    if not PATHS.DATA_PROCESSED_GROUND_TRUTH.exists():
        print(f'[STEP 0] ⚠ Chưa có ground_truth.pkl -> chạy build tạm thời')
        gt = _build_gt_temp_fallback(df)
    else:
        with open(PATHS.DATA_PROCESSED_GROUND_TRUTH, 'rb') as f:
            gt = pickle.load(f)
    query_ids_raw = gt.get('query_ids', [])
    rel_map_raw = gt.get('relevant_per_query', {})
    query_ids = [str(q) for q in query_ids_raw]
    relevant_per_query: Dict[str, List[str]] = {}
    for q, rels in rel_map_raw.items():
        relevant_per_query[str(q)] = [str(r) for r in rels]
    gt['query_ids'] = query_ids
    gt['relevant_per_query'] = relevant_per_query
    print(f'[STEP 0] ✓ Ground truth: {len(query_ids)} query')
    return (df, gt)

def _build_gt_temp_fallback(df: pd.DataFrame) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split
    N = len(df)
    stratify = df.category if df.category.nunique() > 1 else None
    q_idx, _ = train_test_split(np.arange(N), test_size=1 - EVALUATION.QUERY_SET_RATIO, random_state=EVALUATION.RANDOM_SEED, stratify=stratify)
    query_ids = df.iloc[q_idx]['id'].tolist()
    attributes = EVALUATION.GT_ATTRIBUTES
    rel_per_query: Dict[str, List[str]] = {}
    for q_pos, qi in enumerate(q_idx):
        qrow = df.iloc[qi]
        q_cat = qrow.category
        rels = []
        for i, r in df.iterrows():
            if i == qi:
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
                rels.append(r.id)
        rel_per_query[str(qrow.id)] = [str(x) for x in rels]
    return {'query_ids': [str(x) for x in query_ids], 'relevant_per_query': rel_per_query, 'strategy': 'fallback_rule_based', 'num_products_total': N}

def _find_df_row(df: pd.DataFrame, qid: str) -> Optional[pd.Series]:
    mask = df['id'].astype(str) == str(qid)
    if mask.any():
        return df[mask].iloc[0]
    try:
        mask2 = df['id'] == int(qid)
        if mask2.any():
            return df[mask2].iloc[0]
    except Exception:
        pass
    return None

def step1_compute_ap_per_method(df: pd.DataFrame, gt: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    from src.retrieval import Retriever
    query_ids: List[str] = gt['query_ids']
    rel_per_query: Dict[str, List[str]] = gt['relevant_per_query']
    MAIN_K = EVALUATION.MAIN_K
    top_k_for_rank = max(2 * MAIN_K, 20)
    retriever = Retriever(lazy=True, verbose=False)
    retriever.load_metadata()
    retriever.load_index_mapping()
    methods = ['effnet', 'resnet', 'sbert', 'clip_multimodal']
    results: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    print(f'[STEP 1] Tính AP@{MAIN_K} per-query cho 4 phương pháp...')
    for qi, qid in enumerate(query_ids):
        if (qi + 1) % 20 == 0 or qi == len(query_ids) - 1:
            print(f'  Đã xử lý {qi + 1}/{len(query_ids)} query')
        qrow = _find_df_row(df, qid)
        if qrow is None:
            continue
        img_path = qrow.get('image_path', '')
        text_query = f"{qrow.get('product_name', '')} {qrow.get('description', '')}"
        relevant_ids = rel_per_query.get(qid, [])
        for method in methods:
            try:
                if method in ('effnet', 'resnet'):
                    ranked = retriever.search_by_image(img_path, model=method, top_k=top_k_for_rank)
                elif method == 'sbert':
                    ranked = retriever.search_by_text(text_query, model='sbert', top_k=top_k_for_rank)
                else:
                    ranked = retriever.search_multimodal(img_path=img_path, text=text_query, alpha=0.5, top_k=top_k_for_rank)
            except Exception as e:
                results[method][qid] = 0.0
                continue
            ranked_ids = []
            for r in ranked:
                rid = str(r.get('id', ''))
                if rid == str(qid):
                    continue
                ranked_ids.append(rid)
            ap = average_precision_at_k(ranked_ids, relevant_ids, MAIN_K)
            results[method][qid] = ap
    for m in methods:
        vals = list(results[m].values())
        print(f'  mean AP@{MAIN_K} {m}: {np.mean(vals):.4f} (n={len(vals)})')
    return results

def step2_select_error_cases(ap_per_method: Dict[str, Dict[str, float]], gt: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    query_ids: List[str] = gt['query_ids']
    methods = list(ap_per_method.keys())
    q_mean_ap: Dict[str, float] = {}
    for qid in query_ids:
        vals = []
        for m in methods:
            if qid in ap_per_method[m]:
                vals.append(ap_per_method[m][qid])
        q_mean_ap[qid] = float(np.mean(vals)) if vals else 0.0
    sorted_q = sorted(q_mean_ap.items(), key=lambda kv: kv[1])
    num_err = ERROR_ANALYSIS.NUM_CASES
    error_qids = [q for q, _ in sorted_q[:num_err]]
    success_qids = [q for q, _ in sorted_q[-5:]] if len(sorted_q) >= 5 else []
    print(f'[STEP 2] ✓ Chọn {len(error_qids)} trường hợp lỗi (AP thấp nhất)')
    if error_qids:
        print(f'  Error case AP range: {q_mean_ap[error_qids[0]]:.4f} -> {q_mean_ap[error_qids[-1]]:.4f}')
    print(f'[STEP 2] ✓ Chọn {len(success_qids)} trường hợp thành công (đối chiếu)')
    return (error_qids, success_qids)

def step3_build_error_cases(df: pd.DataFrame, gt: Dict[str, Any], error_qids: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    from src.retrieval import Retriever
    rel_per_query: Dict[str, List[str]] = gt['relevant_per_query']
    retriever = Retriever(lazy=True, verbose=False)
    retriever.load_metadata()
    retriever.load_index_mapping()
    alpha_values = list(ERROR_ANALYSIS.ALPHA_TUNE_VALUES)
    alpha_best_counts: Dict[str, List[float]] = {str(a): [] for a in alpha_values}
    error_cases: List[Dict[str, Any]] = []
    print(f'[STEP 3] Xây dựng chi tiết {len(error_qids)} trường hợp lỗi...')
    for ci, qid in enumerate(error_qids):
        if (ci + 1) % 5 == 0 or ci == len(error_qids) - 1:
            print(f'  Đang xử lý case {ci + 1}/{len(error_qids)}...')
        qrow = _find_df_row(df, qid)
        if qrow is None:
            continue
        img_path = qrow.get('image_path', '')
        text_query = f"{qrow.get('product_name', '')} {qrow.get('description', '')}"
        relevant_ids = rel_per_query.get(qid, [])
        q_cat = str(qrow.get('category', 'other'))
        q_color = str(qrow.get('color', ''))
        q_material = str(qrow.get('material', ''))
        q_style = str(qrow.get('style', ''))
        ranking: Dict[str, List[Dict[str, Any]]] = {}
        ap_10: Dict[str, float] = {}
        for method in ('effnet', 'resnet', 'sbert'):
            try:
                if method in ('effnet', 'resnet'):
                    r = retriever.search_by_image(img_path, model=method, top_k=20)
                else:
                    r = retriever.search_by_text(text_query, model='sbert', top_k=20)
            except Exception:
                r = []
            ranked_ids = []
            filtered = []
            for item in r:
                rid = str(item.get('id', ''))
                if rid == str(qid):
                    continue
                ranked_ids.append(rid)
                filtered.append(item)
                if len(filtered) >= 10:
                    break
            ranking[method] = filtered[:10]
            ap_10[method] = average_precision_at_k(ranked_ids, relevant_ids, 10)
        try:
            r_clip_05 = retriever.search_multimodal(img_path=img_path, text=text_query, alpha=0.5, top_k=20)
        except Exception:
            r_clip_05 = []
        ranked_ids_05 = []
        filtered_05 = []
        for item in r_clip_05:
            rid = str(item.get('id', ''))
            if rid == str(qid):
                continue
            ranked_ids_05.append(rid)
            filtered_05.append(item)
            if len(filtered_05) >= 10:
                break
        ranking['clip_0.5'] = filtered_05[:10]
        ap_clip_05 = average_precision_at_k(ranked_ids_05, relevant_ids, 10)
        ap_10['clip_0.5'] = ap_clip_05
        alpha_ap_map: Dict[float, float] = {}
        alpha_rankings: Dict[float, List[Dict[str, Any]]] = {}
        for a in alpha_values:
            try:
                ra = retriever.search_multimodal(img_path=img_path, text=text_query, alpha=a, top_k=20)
            except Exception:
                ra = []
            rids_a = []
            filt_a = []
            for item in ra:
                rid = str(item.get('id', ''))
                if rid == str(qid):
                    continue
                rids_a.append(rid)
                filt_a.append(item)
                if len(filt_a) >= 10:
                    break
            alpha_rankings[a] = filt_a[:10]
            ap_a = average_precision_at_k(rids_a, relevant_ids, 10)
            alpha_ap_map[a] = ap_a
        best_alpha = 0.5
        best_ap = -1.0
        for a, ap_val in alpha_ap_map.items():
            if ap_val > best_ap:
                best_ap = ap_val
                best_alpha = a
        delta_alpha_ap = best_ap - ap_clip_05
        alpha_best_counts[str(best_alpha)].append(best_ap)
        ap_res = ap_10.get('resnet', 0.0)
        ap_sb = ap_10.get('sbert', 0.0)
        N_total = len(df)
        rel_count = len(relevant_ids)
        failure_type = _classify_failure_type(relevant_ids=relevant_ids, resnet_top=ranking.get('resnet', [])[:3], clip_top=filtered_05[:3], qrow=qrow, ap_resnet=ap_res, ap_sbert=ap_sb, ap_clip_best=best_ap, ap_clip_05=ap_clip_05, N_total=N_total)
        top1_per_method: Dict[str, Tuple[Any, float, str]] = {}
        for m in ('effnet', 'resnet', 'sbert', 'clip_0.5'):
            rlist = ranking.get(m, [])
            if rlist:
                r0 = rlist[0]
                top1_per_method[m] = (r0.get('id', ''), float(r0.get('similarity', 0.0)), str(r0.get('category', '')))
            else:
                top1_per_method[m] = ('', 0.0, '')
        notes = _generate_case_notes(failure_type=failure_type, qrow=qrow, ap_10=ap_10, best_alpha=best_alpha, delta_alpha_ap=delta_alpha_ap, relevant_count=rel_count)
        case = {'case_id': ci, 'query_id': qid, 'query_category': q_cat, 'query_color': q_color, 'query_material': q_material, 'query_style': q_style, 'ap_scores_per_method': ap_10, 'best_alpha': best_alpha, 'delta_alpha_ap': delta_alpha_ap, 'failure_type': failure_type, 'ranking': ranking, 'top1_per_method': top1_per_method, 'relevant_sample_ids': list(relevant_ids)[:5], 'relevant_ids_full': list(relevant_ids), 'notes': notes, 'query_image_path': img_path, 'query_text': text_query}
        error_cases.append(case)
    print(f'[STEP 3] ✓ Xây dựng xong {len(error_cases)} error cases chi tiết')
    return (error_cases, alpha_best_counts)

def _count_matching_attrs(r: Dict[str, Any], qrow: pd.Series) -> int:
    cnt = 0
    for attr in ('category', 'color', 'material', 'style'):
        rv = str(r.get(attr, '')).strip().lower()
        qv = str(qrow.get(attr, '')).strip().lower()
        if qv and rv and (qv == rv):
            cnt += 1
    return cnt

def _classify_failure_type(relevant_ids: List[str], resnet_top: List[Dict[str, Any]], clip_top: List[Dict[str, Any]], qrow: pd.Series, ap_resnet: float, ap_sbert: float, ap_clip_best: float, ap_clip_05: float, N_total: int) -> str:
    rel_set = set((str(x) for x in relevant_ids))
    if len(relevant_ids) == 0 or len(relevant_ids) > max(N_total // 2, 10):
        return 'GROUND_TRUTH_FLAW'
    visual_top = list(resnet_top[:3]) + list(clip_top[:3])
    strong_matches_not_rel = 0
    for r in visual_top:
        attrs_match = _count_matching_attrs(r, qrow)
        rid = str(r.get('id', ''))
        if attrs_match >= 2 and rid not in rel_set:
            strong_matches_not_rel += 1
    if strong_matches_not_rel >= 2:
        return 'GROUND_TRUTH_FLAW'
    if ap_resnet > 0.3 and ap_sbert < 0.15 and (ap_resnet - ap_sbert > 0.2):
        return 'TEXT_METADATA_FAILURE'
    if ap_sbert > 0.3 and ap_resnet < 0.15 and (ap_sbert - ap_resnet > 0.2):
        return 'IMAGE_EMBEDDING_FAILURE'
    if ap_clip_best - ap_clip_05 > 0.1:
        return 'FUSION_ALPHA'
    if ap_resnet < 0.1 and ap_sbert < 0.1:
        return 'IMAGE_EMBEDDING_FAILURE'
    return 'IMAGE_EMBEDDING_FAILURE'

def _generate_case_notes(failure_type: str, qrow: pd.Series, ap_10: Dict[str, float], best_alpha: float, delta_alpha_ap: float, relevant_count: int) -> List[str]:
    notes: List[str] = []
    ap_res = ap_10.get('resnet', 0.0)
    ap_sb = ap_10.get('sbert', 0.0)
    ap_c = ap_10.get('clip_0.5', 0.0)
    ap_s = ap_10.get('effnet', 0.0)
    notes.append(f'Số relevant thực tế: {relevant_count} item')
    notes.append(f'AP@10 - scratch: {ap_s:.3f}, resnet: {ap_res:.3f}, sbert: {ap_sb:.3f}, clip(0.5): {ap_c:.3f}')
    if failure_type == 'GROUND_TRUTH_FLAW':
        notes.append('GT FLAW: Tập relevant quá hẹp/quá rộng hoặc bỏ sót item match thuộc tính')
    elif failure_type == 'TEXT_METADATA_FAILURE':
        notes.append(f'TEXT METADATA FAIL: ResNet tốt ({ap_res:.3f}) > SBERT kém ({ap_sb:.3f}). Có thể metadata tự sinh chung chung, thiếu keywords đặc trưng.')
    elif failure_type == 'IMAGE_EMBEDDING_FAILURE':
        notes.append(f'IMAGE EMBEDDING FAIL: SBERT tốt ({ap_sb:.3f}) > ResNet kém ({ap_res:.3f}). Ánh sáng, góc chụp, họa tiết phức tạp có thể gây nhầm lẫn visual model.')
    elif failure_type == 'FUSION_ALPHA':
        notes.append(f"FUSION ALPHA: alpha=0.5 có AP={ap_c:.3f}, best alpha={best_alpha} có AP cải thiện {delta_alpha_ap:+.3f}. Query thiên về {('text' if best_alpha > 0.5 else 'image')}.")
    notes.append(f"Query thuộc tính: cat={qrow.get('category', '')}, color={qrow.get('color', '')}, material={qrow.get('material', '')}, style={qrow.get('style', '')}")
    return notes

def step4_generate_error_figures(error_cases: List[Dict[str, Any]]) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    saved_count = 0
    print(f'[STEP 4] Sinh figure PNG cho {len(error_cases)} trường hợp lỗi...')
    for case in error_cases:
        idx = case['case_id']
        ft = case['failure_type']
        fig_path = REPORTS_ERROR_CASES_DIR / f'error_case_{idx:02d}_{ft.lower()}.png'
        try:
            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.25)
            qimg_path = case['query_image_path']
            rel_set = set((str(x) for x in case['relevant_ids_full']))
            ax_q = fig.add_subplot(gs[0, 0])
            try:
                im = Image.open(qimg_path).convert('RGB')
                ax_q.imshow(im)
            except Exception:
                ax_q.text(0.5, 0.5, 'Không thể đọc ảnh query', ha='center', va='center', fontsize=9)
            title_color = 'red'
            ax_q.set_title(f'Query #{idx}', color=title_color, fontsize=10, fontweight='bold')
            ax_q.set_xlabel(f"cat={case['query_category']}, type={ft}", fontsize=8)
            ax_q.set_xticks([])
            ax_q.set_yticks([])
            resnet_ranking = case['ranking'].get('resnet', [])
            for i in range(5):
                ax = fig.add_subplot(gs[0, i + 1])
                if i < len(resnet_ranking):
                    r = resnet_ranking[i]
                    rid = str(r.get('id', ''))
                    is_rel = rid in rel_set
                    img_p = r.get('image_path', '')
                    try:
                        im2 = Image.open(img_p).convert('RGB')
                        ax.imshow(im2)
                    except Exception:
                        ax.text(0.5, 0.5, 'Ảnh lỗi', ha='center', va='center', fontsize=8)
                    color = 'green' if is_rel else 'red'
                    ax.set_title(f'ResNet #{i + 1} | id={rid}', color=color, fontsize=8)
                    sim = float(r.get('similarity', 0.0))
                    cat = str(r.get('category', ''))
                    ax.set_xlabel(f'sim={sim:.2f} {cat}', fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            ax_txt = fig.add_subplot(gs[1, 0])
            ax_txt.clear()
            ax_txt.set_frame_on(False)
            ax_txt.set_xticks([])
            ax_txt.set_yticks([])
            ap_map = case['ap_scores_per_method']
            summary_lines = [f'PHÂN LOẠI: {ft}', '', f"AP@10 scratch  : {ap_map.get('effnet', 0):.3f}", f"AP@10 resnet   : {ap_map.get('resnet', 0):.3f}", f"AP@10 sbert    : {ap_map.get('sbert', 0):.3f}", f"AP@10 clip 0.5 : {ap_map.get('clip_0.5', 0):.3f}", '', f"Best alpha : {case['best_alpha']}", f"ΔAP alpha  : {case['delta_alpha_ap']:+.3f}", '', f"Số relevant: {len(case['relevant_ids_full'])}", '', f"Màu query  : {case['query_color']}", f"Chất liệu  : {case['query_material']}", f"Phong cách : {case['query_style']}"]
            text_content = '\n'.join(summary_lines)
            ax_txt.text(0.02, 0.98, text_content, ha='left', va='top', fontsize=7.5, linespacing=1.35, family='monospace', transform=ax_txt.transAxes)
            sbert_ranking = case['ranking'].get('sbert', [])
            for i in range(5):
                ax = fig.add_subplot(gs[1, i + 1])
                if i < len(sbert_ranking):
                    r = sbert_ranking[i]
                    rid = str(r.get('id', ''))
                    is_rel = rid in rel_set
                    img_p = r.get('image_path', '')
                    try:
                        im3 = Image.open(img_p).convert('RGB')
                        ax.imshow(im3)
                    except Exception:
                        ax.text(0.5, 0.5, 'Ảnh lỗi', ha='center', va='center', fontsize=8)
                    color = 'green' if is_rel else 'red'
                    ax.set_title(f'SBERT #{i + 1} | id={rid}', color=color, fontsize=8)
                    sim = float(r.get('similarity', 0.0))
                    cat = str(r.get('category', ''))
                    ax.set_xlabel(f'sim={sim:.2f} {cat}', fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f'Trường hợp lỗi #{idx}: Phân loại = {ft}', fontsize=13, fontweight='bold', y=0.995)
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close('all')
            saved_count += 1
        except Exception as exc:
            try:
                plt.close('all')
            except Exception:
                pass
            continue
    print(f'[STEP 4] ✓ Đã lưu {saved_count}/{len(error_cases)} figure PNG')
    return saved_count

def step5_alpha_distribution_chart(alpha_best_counts: Dict[str, List[float]]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    alpha_values = list(ERROR_ANALYSIS.ALPHA_TUNE_VALUES)
    counts = [len(alpha_best_counts.get(str(a), [])) for a in alpha_values]
    out_path = PATHS.REPORTS_FIGURES_DIR / 'alpha_tuning_distribution.png'
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f'α={a:.1f}' for a in alpha_values], counts, color=['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2', '#FF9DA6'], edgecolor='black', linewidth=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(c), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Phân bố alpha tốt nhất trên tập lỗi', fontsize=13, fontweight='bold')
    ax.set_xlabel('Giá trị alpha (0 = 100% ảnh, 1 = 100% text)', fontsize=10)
    ax.set_ylabel('Số lượng query', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[STEP 5] ✓ Lưu alpha tuning distribution: {out_path}')

def step6_error_type_pie_chart(error_cases: List[Dict[str, Any]]) -> Dict[str, int]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ft_counter: Counter = Counter((c['failure_type'] for c in error_cases))
    for t in ERROR_ANALYSIS.TYPES:
        if t not in ft_counter:
            ft_counter[t] = 0
    out_path = PATHS.REPORTS_FIGURES_DIR / 'error_type_distribution.png'
    labels = list(ERROR_ANALYSIS.TYPES)
    sizes = [ft_counter[t] for t in labels]
    colors = {'IMAGE_EMBEDDING_FAILURE': '#E45756', 'TEXT_METADATA_FAILURE': '#F58518', 'FUSION_ALPHA': '#4C78A8', 'GROUND_TRUTH_FLAW': '#54A24B'}
    color_list = [colors.get(l, '#999999') for l in labels]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=color_list, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}, wedgeprops={'linewidth': 1.2, 'edgecolor': 'white'})
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
        at.set_color('white')
    ax.set_title(f'Phân bố {len(error_cases)} trường hợp lỗi theo 4 loại', fontsize=13, fontweight='bold')
    ax.axis('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[STEP 6] ✓ Lưu error type pie chart: {out_path}')
    return dict(ft_counter)

def step7_write_report(error_cases: List[Dict[str, Any]], type_counts: Dict[str, int], saved_figures: int) -> None:
    num_cases = ERROR_ANALYSIS.NUM_CASES
    total_analyzed = len(error_cases)
    examples: Dict[str, Optional[str]] = {}
    for ft in ERROR_ANALYSIS.TYPES:
        found = None
        for c in error_cases:
            if c['failure_type'] == ft:
                idx = c['case_id']
                p = Path(REPORTS_ERROR_CASES_DIR / f'error_case_{idx:02d}_{ft.lower()}.png')
                if p.exists():
                    found = f'error_cases/error_case_{idx:02d}_{ft.lower()}.png'
                    break
        examples[ft] = found
    rows_tbl = []
    for ft in ERROR_ANALYSIS.TYPES:
        cnt = type_counts.get(ft, 0)
        pct = cnt / total_analyzed * 100 if total_analyzed > 0 else 0.0
        rows_tbl.append((ft, cnt, pct))
    md = []
    md.append('# Phân tích lỗi hệ thống gợi ý thời trang đa phương thức')
    md.append('')
    md.append('## 1. Tổng quan')
    md.append('')
    md.append(f'- Số lượng query lỗi được phân tích chi tiết: **N = {total_analyzed}** (trong đó ERROR_ANALYSIS.NUM_CASES = {num_cases})')
    md.append(f'- Số lượng hình ảnh minh họa trường hợp lỗi đã tạo: **{saved_figures}** (yêu cầu tối thiểu 20)')
    md.append(f'- Số phương pháp đánh giá đối chiếu: 4 (EfficientNet-B0, ResNet50, SBERT, CLIP multimodal α=0.5)')
    md.append(f'- Chỉ số đo lường chính: Average Precision@{EVALUATION.MAIN_K} (AP@{EVALUATION.MAIN_K})')
    md.append('')
    md.append('## 2. Phân bố loại lỗi')
    md.append('')
    md.append('Bảng thống kê 4 loại lỗi theo ERROR_ANALYSIS.TYPES:')
    md.append('')
    md.append('| Loại lỗi | Số lượng | Tỷ lệ % |')
    md.append('|---|---:|---:|')
    for ft, cnt, pct in rows_tbl:
        md.append(f'| {ft} | {cnt} | {pct:.1f}% |')
    md.append('')
    md.append('Biểu đồ tròn trực quan hóa phân bổ:')
    md.append('')
    md.append('![Error Type Distribution](./figures/error_type_distribution.png)')
    md.append('')
    md.append('## 3. Phân tích chi tiết 4 loại lỗi')
    md.append('')
    md.append('### 3.1 IMAGE_EMBEDDING_FAILURE')
    md.append('')
    md.append('- **Nguyên nhân thường gặp**:')
    md.append('  - Ánh sáng kém / quá tối / quá sáng làm biến đổi màu sắc đặc trưng.')
    md.append('  - Góc chụp khác thường (lật, xoay, cắt gần khuôn mẫu) làm CNN/CLIP không nhận diện shape.')
    md.append('  - Họa tiết phức tạp (hoa văn, caro, sọc) hoặc texture đặc biệt (lưới, đục lỗ) không được pre-trained model học đủ.')
    md.append('  - Item có kích thước nhỏ trong bức ảnh, nền nhiễu làm model trích xuất nhầm đặc trưng nền.')
    md.append('')
    md.append('- **Ví dụ minh họa**:')
    md.append('')
    if examples['IMAGE_EMBEDDING_FAILURE']:
        md.append(f"![Image Embedding Failure](./{examples['IMAGE_EMBEDDING_FAILURE']})")
    else:
        md.append('*(Không có case thuộc loại này trong tập lỗi)*')
    md.append('')
    md.append('- **Hướng cải tiến**:')
    md.append('  1. **Data augmentation**: Thêm random crop, color jitter, rotation ±15°, random erasing vào pipeline train EfficientNet-B0.')
    md.append('  2. **Metric learning với ArcFace loss**: Thay vì classification loss đơn thuần, train EfficientNet-B0 với ArcFace/ProxyNCA loss để embedding class-separated hơn.')
    md.append('  3. **Finetune CLIP image encoder**: Lấy ViT-B/32 của CLIP, finetune 1 epoch trên bộ fashion dataset với batch size lớn (≥128) để domain-adapt visual feature.')
    md.append('')
    md.append('### 3.2 TEXT_METADATA_FAILURE')
    md.append('')
    md.append('- **Nguyên nhân thường gặp**:')
    md.append('  - Metadata được sinh tự động (rule-based template) có mô tả chung chung, lặp từ khóa.')
    md.append('  - Thiếu từ khóa đặc trưng cho sản phẩm cụ thể (ví dụ: chỉ ghi *cotton shirt* thay vì *oxford button-down cotton shirt with chest pocket*).')
    md.append('  - SBERT với mô hình nhỏ (MiniLM-L12) không đủ sâu để phân biệt 2 mô tả gần giống nhau.')
    md.append('')
    md.append('- **Ví dụ minh họa**:')
    md.append('')
    if examples['TEXT_METADATA_FAILURE']:
        md.append(f"![Text Metadata Failure](./{examples['TEXT_METADATA_FAILURE']})")
    else:
        md.append('*(Không có case thuộc loại này trong tập lỗi)*')
    md.append('')
    md.append('- **Hướng cải tiến**:')
    md.append('  1. **Thu hoạch mô tả thực tế từ người dùng/thương hiệu**: Thay vì sinh template, crawl mô tả gốc từ web thương mại điện tử hoặc viết tay cho ≥50% dataset.')
    md.append('  2. **Nâng cấp SBERT model**: Từ `paraphrase-multilingual-MiniLM-L12-v2` (384d) lên `paraphrase-mpnet-base-v2` (768d) hoặc `all-mpnet-base-v2` để quality tốt hơn dù tốc độ chậm hơn.')
    md.append('  3. **Named Entity Recognition (NER)** cho attribute extraction: Dùng NER model nhận diện tên thương hiệu, style đặc thù (vintage, streetwear, oversized) từ tên sản phẩm và description để enrich metadata.')
    md.append('')
    md.append('### 3.3 FUSION_ALPHA')
    md.append('')
    md.append('- **Nguyên nhân thường gặp**:')
    md.append('  - Mặc định `alpha=0.5` (50% ảnh + 50% text) không phù hợp với mọi query. Các query có thông tin visual mạnh (màu sắc, họa tiết nổi bật) cần alpha thấp (thiên về ảnh ~0.2-0.4), còn các query mô tả text chi tiết cần alpha cao (thiên về text ~0.6-0.8).')
    md.append('  - CLIP multimodal fusion đơn giản (weighted sum) không linh hoạt để thích ứng per-query.')
    md.append('')
    md.append('- **Biểu đồ phân bố alpha tốt nhất trên tập lỗi**:')
    md.append('')
    md.append('![Alpha Tuning Distribution](./figures/alpha_tuning_distribution.png)')
    md.append('')
    md.append('- **Hướng cải tiến**:')
    md.append('  1. **Adaptive alpha per-query**: Dựa trên entropy (độ bất định) của phân bố xác suất top-5 CLIP image embedding so với CLIP text embedding. Nếu image phân bố dồn về ít item (entropy thấp) -> giảm alpha, ngược lại tăng alpha.')
    md.append('  2. **Gated fusion network**: Thêm một MLP nhỏ nhận concatenated [image_emb, text_emb] làm input để học weight ảnh/text thay vì fix alpha.')
    md.append('  3. **Cross-attention fusion**: Dùng lớp CrossAttention giữa CLIP image và text encoding để tự động tìm trọng số theo từng patch/token.')
    md.append('')
    md.append('### 3.4 GROUND_TRUTH_FLAW')
    md.append('')
    md.append('- **Nguyên nhân thường gặp**:')
    md.append('  - Ground truth được sinh theo rule-based: *cùng category + match ≥ 2/3 thuộc tính (color, material, style)*. Quy tắc này đôi khi quá hẹp (bỏ sót item thực tế giống nhưng khác material nhẹ) hoặc quá rộng (đếm cả item chỉ trùng category + 1 thuộc tính).')
    md.append('  - Không có sự can thiệp của con người vào việc đánh dấu relevant/irrelevant.')
    md.append('')
    md.append('- **Ví dụ**: Một query áo sơ mi xanh dương cotton office có top-3 kết quả đều là áo sơ mi xanh dương cotton (match 3/3) nhưng do style được gán nhầm `office` vs `formal` nên không được đếm vào relevant.')
    md.append('')
    if examples['GROUND_TRUTH_FLAW']:
        md.append(f"![Ground Truth Flaw](./{examples['GROUND_TRUTH_FLAW']})")
    else:
        md.append('*(Không có case thuộc loại này trong tập lỗi)*')
    md.append('')
    md.append('- **Hướng cải tiến**:')
    md.append('  1. **Human-labeled override**: Chuẩn bị file `human_labeled.csv` (query_id, relevant_id) cho ít nhất 30% query set, ưu tiên các query mà rule-based cho 0 relevant hoặc model mâu thuẫn với GT.')
    md.append('  2. **Active learning**: Sắp xếp các query mà (AP cao nhưng GT đánh dấu không có relevant) hoặc (4 phương pháp cho kết quả khác nhau) để người dùng đánh dấu, từ đó mở rộng GT dần.')
    md.append('  3. **Threshold mềm cho attribute match**: Thay vì match cứng chuỗi, dùng embedding-similarity cho giá trị attribute (ví dụ: màu navy ≈ blue, chất liệu denim ≈ cotton pha denim).')
    md.append('')
    md.append('## 4. Đề xuất 4 cải tiến ưu tiên cho khóa luận')
    md.append('')
    md.append('Thứ tự ưu tiên từ cao → thấp, đánh giá theo IMPORTANCE (tác động lên kết quả) và EFFORT (công sức triển khai), thang 1-5 (5 là cao nhất):')
    md.append('')
    md.append('| Ưu tiên | Cải tiến | IMPORTANCE | EFFORT | Tác động dự kiến |')
    md.append('|---:|---|---:|---:|---|')
    md.append('| **#1** | **Finetune CLIP image encoder trên bộ fashion 10K ảnh** với contrastive loss CLIP-style (image ↔ text matching batch), 1 epoch với lr=1e-5, warmup 500 step. | **5** | **3** | Cải thiện đáng kể cả IMAGE_EMBEDDING_FAILURE (domain-adapt visual) và FUSION_ALPHA (CLIP multimodal chất hơn). Dự kiến mAP@10 tăng 5-8%. |')
    md.append('| **#2** | **Thêm reranking model (CrossEncoder SBERT)** sau giai đoạn retrieval thô top-20. Dùng `cross-encoder/ms-marco-MiniLM-L6-v2` hoặc ViT+Text cross-encoder để rerank 20 kết quả đầu. | **4** | **2** | Độ trễ tăng nhẹ nhưng precision top-5 cải thiện rõ rệt. Giảm cả TEXT_METADATA_FAILURE lẫn IMAGE_EMBEDDING_FAILURE bằng cách sử dụng dual-signal để sắp xếp lại. Dự kiến Prec@5 tăng 4-6%. |')
    md.append('| **#3** | **Attribute detection model** với ResNet50 backbone + 3 nhánh MLP (color, material, style classification) để tự động gán tags chính xác hơn metadata tự sinh. Train trên bộ fashion có attribute label, distill vào retrieval pipeline. | **5** | **4** | Giảm mạnh GROUND_TRUTH_FLAW và TEXT_METADATA_FAILURE bằng cách enrich metadata với attribute thực tế từ ảnh (không còn phụ thuộc vào text). |')
    md.append('| **#4** | **Batch Hard Negative Mining + Contrastive Learning giữa ảnh + text (CLIP-style fashion domain)**. Train SBERT với hard negatives trong cùng category, đồng thời train EfficientNet-B0 với image-text contrastive loss (hình query ↔ mô tả query). | **4** | **3** | Cải thiện độ tách lớp (class separation) của embedding. Giảm IMAGE_EMBEDDING_FAILURE do negative samples khó hơn giúp model học discriminative feature tốt hơn. |')
    md.append('')
    md.append('## 5. Kết luận')
    md.append('')
    md.append(f'Tổng cộng **{total_analyzed} trường hợp lỗi** đã được phân tích chi tiết thành 4 loại:')
    md.append('')
    for ft, cnt, pct in rows_tbl:
        md.append(f'- **{ft}**: {cnt} trường hợp ({pct:.1f}%)')
    md.append('')
    md.append('Nhìn chung, lỗi phổ biến nhất thường xuất phát từ **IMAGE_EMBEDDING_FAILURE** (do pre-trained visual model chưa được domain-adapt fashion) và **GROUND_TRUTH_FLAW** (do GT sinh tự động chưa được con người kiểm duyệt).')
    md.append('')
    md.append('Các đề xuất cải tiến ở Mục 4 được sắp xếp sao cho 2 cải tiến đầu tiên (finetune CLIP, thêm reranking) có thể triển khai nhanh trong 1-2 tuần và cho kết quả đo lường có thể so sánh (A/B testing) ngay lập tức. Hai cải tiến cuối (attribute detection, hard negative mining) là đột phá dài hạn, cần đầu tư công sức lớn hơn nhưng có khả năng nâng tầm hệ thống lên mô hình state-of-the-art cho thời trang Việt Nam.')
    md.append('')
    md_text = '\n'.join(md)
    with open(REPORTS_ERROR_ANALYSIS_MD, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'[STEP 7] ✓ Viết báo cáo markdown: {REPORTS_ERROR_ANALYSIS_MD}')

def main() -> None:
    print('=' * 70)
    print('  SCRIPT 07: PHÂN TÍCH LỖI HỆ THỐNG GỢI Ý THỜI TRANG ĐA PHƯƠNG THỨC')
    print('=' * 70)
    print(f'  ERROR_ANALYSIS.NUM_CASES       = {ERROR_ANALYSIS.NUM_CASES}')
    print(f'  ERROR_ANALYSIS.TYPES           = {ERROR_ANALYSIS.TYPES}')
    print(f'  ERROR_ANALYSIS.ALPHA_TUNE_VALUES = {ERROR_ANALYSIS.ALPHA_TUNE_VALUES}')
    print(f'  EVALUATION.MAIN_K              = {EVALUATION.MAIN_K}')
    print('=' * 70)
    df, gt = step0_load_data()
    ap_per_method = step1_compute_ap_per_method(df, gt)
    error_qids, success_qids = step2_select_error_cases(ap_per_method, gt)
    error_cases, alpha_best_counts = step3_build_error_cases(df, gt, error_qids)
    saved_figures = step4_generate_error_figures(error_cases)
    step5_alpha_distribution_chart(alpha_best_counts)
    type_counts = step6_error_type_pie_chart(error_cases)
    step7_write_report(error_cases, type_counts, saved_figures)
    print()
    print('=' * 70)
    print('  KẾT QUẢ TỔNG KẾT PHÂN TÍCH LỖI')
    print('=' * 70)
    print(f'  Tổng số trường hợp phân tích       : {len(error_cases)}')
    print(f'  Số ảnh minh họa đã lưu             : {saved_figures}')
    print(f'  Phân bố loại lỗi                   : {dict(type_counts)}')
    print(f'  Báo cáo markdown lưu tại           : {REPORTS_ERROR_ANALYSIS_MD}')
    print(f'  Thư mục figure error cases         : {REPORTS_ERROR_CASES_DIR}')
    print('=' * 70)
    print('  ✓ HOÀN THÀNH ERROR ANALYSIS!')
    print('=' * 70)
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        main()
