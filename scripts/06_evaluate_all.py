import argparse, json, os, pickle, sys, warnings
PROJECT_ROOT_TMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_TMP not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TMP)
import _bootstrap_env
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from config import PATHS, EVALUATION

def _check_scratch_zeros(retriever) -> bool:
    try:
        if PATHS.EMB_MANIFEST.exists():
            with open(PATHS.EMB_MANIFEST, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            scratch_info = manifest.get('scratch_cnn', {})
            if isinstance(scratch_info, dict):
                mean_norm = scratch_info.get('mean_norm', None)
                if mean_norm is not None and float(mean_norm) < 1e-06:
                    return True
        try:
            emb = retriever.get_embedding('scratch_cnn')
            mean_norm = float(np.mean(np.linalg.norm(emb, axis=1)))
            if mean_norm < 1e-06:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False

def _draw_charts(metrics_df: pd.DataFrame, bar_path: Path, curve_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            sns.set_style('whitegrid')
            use_seaborn = True
        except ImportError:
            use_seaborn = False
    except Exception as e:
        print(f'⚠️  Không thể import matplotlib/biểu đồ sẽ bị bỏ qua: {e}')
        return
    try:
        k_list_sorted = sorted(EVALUATION.K_LIST)
        methods = metrics_df['method'].tolist()
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        recall_5 = metrics_df['Recall@5'].tolist()
        recall_10 = metrics_df['Recall@10'].tolist()
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax1.bar(x - width / 2, recall_5, width, label='Recall@5', color='#4C72B0')
        bars2 = ax1.bar(x + width / 2, recall_10, width, label='Recall@10', color='#DD8452')
        ax1.set_xlabel('Mô hình', fontsize=11)
        ax1.set_ylabel('Giá trị', fontsize=11)
        ax1.set_title('Recall@K theo mô hình', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax1.legend()
        ax1.set_ylim(0, max(max(recall_5), max(recall_10)) * 1.25 if max(max(recall_5), max(recall_10)) > 0 else 1.0)
        ax1.grid(axis='y', alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        map_10 = metrics_df[f'mAP@{EVALUATION.MAIN_K}'].tolist()
        f1_5 = metrics_df['F1@5'].tolist()
        bars3 = ax2.bar(x - width / 2, map_10, width, label=f'mAP@{EVALUATION.MAIN_K}', color='#55A868')
        bars4 = ax2.bar(x + width / 2, f1_5, width, label='F1@5', color='#C44E52')
        ax2.set_xlabel('Mô hình', fontsize=11)
        ax2.set_ylabel('Giá trị', fontsize=11)
        ax2.set_title(f'mAP@{EVALUATION.MAIN_K} và F1@5 theo mô hình', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax2.legend()
        ax2.set_ylim(0, max(max(map_10), max(f1_5)) * 1.25 if max(max(map_10), max(f1_5)) > 0 else 1.0)
        ax2.grid(axis='y', alpha=0.3)
        for bars in [bars3, bars4]:
            for bar in bars:
                h = bar.get_height()
                ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        fig1.tight_layout()
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(bar_path, dpi=120, bbox_inches='tight')
        plt.close(fig1)
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
        markers = ['o', 's', '^', 'D', 'v', '*']
        for idx, method in enumerate(methods):
            row = metrics_df[metrics_df['method'] == method].iloc[0]
            recalls = [row[f'Recall@{k}'] for k in k_list_sorted]
            ax3.plot(k_list_sorted, recalls, label=method, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], linewidth=2, markersize=7)
        ax3.set_xlabel('K', fontsize=11)
        ax3.set_ylabel('Recall@K', fontsize=11)
        ax3.set_title('Đường Recall@K theo mô hình', fontsize=13, fontweight='bold')
        ax3.set_xticks(k_list_sorted)
        ax3.set_ylim(0, 1.0)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(curve_path, dpi=120, bbox_inches='tight')
        plt.close(fig2)
    except Exception as e:
        print(f'⚠️  Lỗi khi vẽ biểu đồ (bỏ qua): {e}')
        try:
            plt.close('all')
        except Exception:
            pass

def main() -> None:
    parser = argparse.ArgumentParser(description='so sánh 6 mô hình retrieval và xuất báo cáo')
    parser.add_argument('--top-k-max', type=int, default=max(EVALUATION.K_LIST), help=f'Top-K tối đa để retrieval (mặc định max(K_LIST)={max(EVALUATION.K_LIST)})')
    parser.add_argument('--outdir', type=str, default=str(PATHS.REPORTS_DIR), help=f'Thư mục xuất báo cáo (mặc định {PATHS.REPORTS_DIR})')
    parser.add_argument('--skip-existing', action='store_true', help='Bỏ qua nếu metrics.csv đã tồn tại')
    args = parser.parse_args()
    PATHS.ensure_all()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir = outdir / 'figures'
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = outdir / 'metrics.csv'
    metrics_json_path = outdir / 'metrics_dict.json'
    bar_chart_path = reports_figures_dir / 'metrics_bar_chart.png'
    recall_curve_path = reports_figures_dir / 'recall_curve.png'
    if args.skip_existing and metrics_csv_path.exists():
        print(f'⚠️  Đã tồn tại {metrics_csv_path} --skip-existing => Bỏ qua')
        return
    print('=' * 70)
    print('Bước 1/6: Tải ground_truth.pkl')
    print('=' * 70)
    if not PATHS.DATA_PROCESSED_GROUND_TRUTH.exists():
        raise FileNotFoundError(f'Không tìm thấy ground_truth.pkl tại {PATHS.DATA_PROCESSED_GROUND_TRUTH}. Chạy scripts/05_build_ground_truth.py trước!')
    with open(PATHS.DATA_PROCESSED_GROUND_TRUTH, 'rb') as f:
        gt = pickle.load(f)
    query_ids = gt['query_ids']
    rel_per_q = {str(k): [str(x) for x in v] for k, v in gt['relevant_per_query'].items()}
    gt_strategy = gt.get('strategy', 'unknown')
    print(f'✓ Đã tải ground truth: strategy={gt_strategy}')
    print(f'✓ Số query: {len(query_ids)}')
    print()
    print('=' * 70)
    print('Bước 2/6: Khởi tạo Retriever')
    print('=' * 70)
    from src.retrieval import Retriever
    from src import metrics as metrics_module
    retriever = Retriever(lazy=False, verbose=True)
    print('✓ Retriever đã khởi tạo và nạp metadata/index mapping')
    print()
    print('=' * 70)
    print('Bước 3/6: Kiểm tra Scratch CNN embedding')
    print('=' * 70)
    if _check_scratch_zeros(retriever):
        print('⚠️  WARNING: Scratch CNN chưa train -> embedding toàn 0 -> chỉ số gần như random. OK để test pipeline.')
    else:
        print('✓ Scratch CNN embedding OK (có norm trung bình > 0)')
    print()
    print('=' * 70)
    print('Bước 4/6: Định nghĩa 6 mô hình so sánh')
    print('=' * 70)
    methods = [('CLIP Multimodal (alpha=0.5)', {'mode': 'multimodal', 'alpha': 0.5}), ('CLIP Multimodal (alpha=0.7)', {'mode': 'multimodal', 'alpha': 0.7}), ('CLIP Image', {'mode': 'image', 'model': 'clip'}), ('ResNet50 ', {'mode': 'image', 'model': 'resnet'}), ('EfficientNet-B0', {'mode': 'image', 'model': 'effnet'}), ('TF-IDF', {'mode': 'text', 'model': 'tfidf'}), ('Sentence-BERT', {'mode': 'text', 'model': 'sbert'})]
    for i, (name, _) in enumerate(methods, 1):
        print(f'  {i}. {name}')
    print()
    print('=' * 70)
    print('Bước 5/6: so sánh từng mô hình')
    print('=' * 70)
    rows = []
    for idx, (method_name, mode_cfg) in enumerate(methods, 1):
        print()
        print(f'--- [{idx}/{len(methods)}] so sánh: {method_name} ---')
        try:
            row = metrics_module.evaluate_method(method_name=method_name, retriever=retriever, query_ids=query_ids, relevant_per_query=rel_per_q, mode_config=mode_cfg, k_list=EVALUATION.K_LIST)
            rows.append(row)
            print(f"  ✓ n_queries={row['n_queries']}, n_valid={row['n_valid_queries']}, Recall@10={row.get('Recall@10', 0):.4f}, mAP@{EVALUATION.MAIN_K}={row.get(f'mAP@{EVALUATION.MAIN_K}', 0):.4f}")
        except Exception as e:
            print(f'  ✗ Lỗi khi so sánh {method_name}: {e}')
            import traceback
            traceback.print_exc()
            empty_row = {'method': method_name, 'n_queries': len(query_ids), 'n_valid_queries': 0}
            for k in EVALUATION.K_LIST:
                empty_row[f'Recall@{k}'] = 0.0
                empty_row[f'Precision@{k}'] = 0.0
                empty_row[f'F1@{k}'] = 0.0
            empty_row[f'mAP@{EVALUATION.MAIN_K}'] = 0.0
            rows.append(empty_row)
    print()
    print('=' * 70)
    print('Bước 6/6: Lưu kết quả & vẽ biểu đồ')
    print('=' * 70)
    metrics_df = pd.DataFrame(rows)
    float_cols = metrics_df.select_dtypes(include=[np.floating, float]).columns
    metrics_df[float_cols] = metrics_df[float_cols].round(4)
    metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8')
    print(f'✓ Đã lưu metrics CSV -> {metrics_csv_path}')
    metrics_dict = {'gt_strategy': gt_strategy, 'k_list': list(EVALUATION.K_LIST), 'main_k': EVALUATION.MAIN_K, 'n_queries_total': len(query_ids), 'created_at': pd.Timestamp.utcnow().isoformat(), 'rows': metrics_df.to_dict(orient='records')}
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    print(f'✓ Đã lưu metrics JSON -> {metrics_json_path}')
    _draw_charts(metrics_df, bar_chart_path, recall_curve_path)
    if bar_chart_path.exists():
        print(f'✓ Đã lưu biểu đồ cột -> {bar_chart_path}')
    if recall_curve_path.exists():
        print(f'✓ Đã lưu đường Recall@K -> {recall_curve_path}')
    print()
    print('=' * 70)
    print('TỔNG KẾT so sánh')
    print('=' * 70)
    display_cols = ['method', 'n_valid_queries']
    for k in [5, 10]:
        if f'Recall@{k}' in metrics_df.columns:
            display_cols.append(f'Recall@{k}')
    for k in [5, 10]:
        if f'Precision@{k}' in metrics_df.columns:
            display_cols.append(f'Precision@{k}')
    if f'mAP@{EVALUATION.MAIN_K}' in metrics_df.columns:
        display_cols.append(f'mAP@{EVALUATION.MAIN_K}')
    if 'F1@5' in metrics_df.columns:
        display_cols.append('F1@5')
    disp_df = metrics_df[display_cols].copy()
    try:
        print(disp_df.to_markdown(index=False, floatfmt='.4f', tablefmt='github'))
    except Exception:
        try:
            from tabulate import tabulate
            print(tabulate(disp_df, headers='keys', showindex=False, floatfmt='.4f', tablefmt='github'))
        except Exception:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(disp_df.to_string(index=False))
    print()
    print('✓ HOÀN THÀNH: so sánh tất cả mô hình xong!')
if __name__ == '__main__':
    main()
