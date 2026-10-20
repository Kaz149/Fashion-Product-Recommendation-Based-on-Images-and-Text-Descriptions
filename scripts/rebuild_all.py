from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
import _bootstrap_env
from config import PATHS, TRAINING, EVALUATION
RED = '\x1b[31m'
GREEN = '\x1b[32m'
YELLOW = '\x1b[33m'
BLUE = '\x1b[34m'
RESET = '\x1b[0m'

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Master Rebuild Pipeline: chạy 6 bước từ chuẩn bị dataset -> đánh giá -> phân tích lỗi')
    ap.add_argument('--dataset-format', '-f', choices=['A', 'B', 'C'], default='C', help='Dataset format: A=folder-per-category, B=CSV+images, C=flat folder auto-predict (default: C)')
    ap.add_argument('--dataset-path', '-p', type=str, default='data/raw/images', help='Path to raw dataset (default: data/raw/images)')
    ap.add_argument('--batch-size-dummy', type=int, default=TRAINING.EPOCHS, help=f'Số epoch train scratch CNN (default: {TRAINING.EPOCHS})')
    ap.add_argument('--batch-size', type=int, default=TRAINING.BATCH_SIZE, help=f'Batch size cho training và embedding (default: {TRAINING.BATCH_SIZE})')
    ap.add_argument('--lr', type=float, default=TRAINING.LEARNING_RATE, help=f'Learning rate scratch CNN (default: {TRAINING.LEARNING_RATE})')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    ap.add_argument('--force', action='store_true', help='Force retrain scratch CNN ngay cả khi checkpoint đã tồn tại')
    ap.add_argument('--skip-train', action='store_true', help='Bỏ qua hoàn toàn step 2 (train scratch CNN). Embedding scratch sẽ dùng zero fallback làm baseline so sánh, hữu ích khi không có GPU.')
    ap.add_argument('--only-steps', type=str, default='', help='Chỉ chạy các bước được chỉ định (danh sách phẩy), ví dụ: --only-steps 1,5,6,7. Mặc định chạy tất cả 6 bước.')
    return ap.parse_args()

def _st_print(color: str, msg: str) -> None:
    print(f'{color}{msg}{RESET}')

def run_step(step_num: int, step_name: str, description: str, cmd: list[str]) -> bool:
    print()
    _st_print(BLUE, '=' * 72)
    _st_print(BLUE, f'[STEP {step_num}/6] {step_name}')
    _st_print(BLUE, f'         {description}')
    _st_print(BLUE, f"         CMD: {' '.join(cmd)}")
    _st_print(BLUE, '=' * 72)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, text=True, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        _st_print(RED, f'[ERROR Step {step_num}] FAILED with code {result.returncode}. Abort pipeline.')
        _st_print(RED, f'         Elapsed: {elapsed:.1f}s')
        return False
    _st_print(GREEN, f'[OK Step {step_num}] COMPLETED in {elapsed:.1f}s.')
    return True

def main() -> int:
    args = parse_args()
    PATHS.ensure_all()
    only_steps: set[int] = set()
    if args.only_steps and args.only_steps.strip():
        for tok in args.only_steps.split(','):
            tok = tok.strip()
            if tok.isdigit():
                n = int(tok)
                if 1 <= n <= 7:
                    only_steps.add(n)

    def should_run(step_num: int) -> bool:
        if only_steps:
            return step_num in only_steps
        return True
    print()
    print('=' * 72)
    print('  🔥 MASTER REBUILD PIPELINE 🔥')
    print('=' * 72)
    print(f'   Dataset format   : {args.dataset_format}')
    print(f'   Dataset path     : {args.dataset_path}')
    print(f'   Batch size       : {args.batch_size}')
    print(f'   Learning rate    : {args.lr}')
    print(f'   Seed             : {args.seed}')
    print(f'   Force retrain    : {args.force}')
    print(f'   Skip train (scr) : {args.skip_train}')
    if only_steps:
        print(f'   Only steps       : {sorted(only_steps)}')
    print('=' * 72)
    print()
    steps_run_count = 0
    t_start = time.time()
    if should_run(1):
        cmd = [sys.executable, 'scripts/01_prepare_dataset.py', '--dataset-format', args.dataset_format, '--dataset-path', args.dataset_path, '--seed', str(args.seed)]
        if not run_step(1, 'Prepare Dataset', 'Chuẩn hóa dataset -> metadata.csv + ảnh resize 224x224', cmd):
            return 1
        steps_run_count += 1
    if should_run(3):
        cmd = [sys.executable, 'scripts/03_extract_all_embeddings.py', '--type', 'all', '--skip-existing', '--batch-size', str(args.batch_size)]
        if not run_step(3, 'Extract All Embeddings', 'Trích xuất 7 loại embedding (4 ảnh + 3 text) -> .npy', cmd):
            return 1
        steps_run_count += 1
    if should_run(4):
        cmd = [sys.executable, 'scripts/04_build_embedding_manifest.py']
        if not run_step(4, 'Build Embedding Manifest', 'Tạo SHA-256 fingerprint, version info -> embedding_manifest.json', cmd):
            return 1
        steps_run_count += 1
    if should_run(5):
        cmd = [sys.executable, 'scripts/05_build_ground_truth.py']
        if not run_step(5, 'Build Ground Truth', 'Rule-based GT + human labeled override CSV -> ground_truth.pkl', cmd):
            return 1
        steps_run_count += 1
    if should_run(6):
        top_k_max = max(EVALUATION.K_LIST)
        cmd = [sys.executable, 'scripts/06_evaluate_all.py', '--skip-existing', '--top-k-max', str(top_k_max)]
        if not run_step(6, 'Evaluate All 6 Methods', 'Đánh giá 6 phương pháp -> metrics.csv + biểu đồ PNG', cmd):
            return 1
        steps_run_count += 1
    if should_run(7):
        cmd = [sys.executable, 'scripts/07_error_analysis.py']
        if not run_step(7, 'Run Error Analysis', 'Phân tích 25 trường hợp lỗi -> 25 PNG + error_analysis.md', cmd):
            return 1
        steps_run_count += 1
    total_elapsed = time.time() - t_start
    print()
    print('=' * 72)
    print(f'  🏁 REBUILD COMPLETE. You can now run: streamlit run app.py')
    print('=' * 72)
    print(f'   Total steps run : {steps_run_count}/7')
    print(f'   Total elapsed   : {total_elapsed:.1f} seconds')
    print()
    if args.skip_train:
        _st_print(YELLOW, '   ⚠️  Lưu ý: Scratch CNN training đã bị SKIP. Embeddings scratch_cnn đang dùng zero fallback.')
        _st_print(YELLOW, '      Để huấn luyện scratch (tốt cho so sánh đầy đủ 6 phương pháp), chạy:')
        _st_print(YELLOW, '      python scripts/02_train_scratch_cnn.py --batch-size-dummy 30 --force')
        _st_print(YELLOW, '      python scripts/03_extract_all_embeddings.py --type image --skip-existing')
    print('=' * 72)
    return 0
if __name__ == '__main__':
    sys.exit(main())
