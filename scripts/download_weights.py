from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow network access for this script even though _bootstrap_env pins offline=1.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
import _bootstrap_env  

from config import TEXT_MODELS, CLIP as CLIP_CFG, DEVICE

SBERT_NAME = TEXT_MODELS.SBERT_NAME
CLIP_NAME = CLIP_CFG.MODEL_NAME
CLIP_PRETRAINED = CLIP_CFG.PRETRAINED


def _cache_ok_sbert() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(SBERT_NAME, device="cpu")
        return m is not None
    except Exception:
        return False


def _cache_ok_clip() -> bool:
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED, device="cpu")
        return model is not None
    except Exception:
        return False


def download_sbert(force: bool = False) -> bool:
    if not force and _cache_ok_sbert():
        print(f"[SBERT] already cached: {SBERT_NAME}")
        return True
    print(f"[SBERT] downloading {SBERT_NAME} ...")
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(SBERT_NAME, device="cpu")
        m.max_seq_length = 128
        # force a forward pass so weights are materialised in cache
        m.encode(["warmup"], show_progress_bar=False)
        print(f"[SBERT] OK (dim={m.get_sentence_embedding_dimension()})")
        return True
    except Exception as exc:
        print(f"[SBERT] FAILED: {exc}")
        return False


def download_clip(force: bool = False) -> bool:
    if not force and _cache_ok_clip():
        print(f"[CLIP] already cached: {CLIP_NAME} ({CLIP_PRETRAINED})")
        return True
    print(f"[CLIP] downloading {CLIP_NAME} ({CLIP_PRETRAINED}) ...")
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED, device="cpu")
        model.eval()
        print("[CLIP] OK")
        return True
    except Exception as exc:
        print(f"[CLIP] FAILED: {exc}")
        return False


def verify_torchvision() -> bool:
    try:
        import torch
        import torchvision
        print(f"[torch] {torch.__version__} | [torchvision] {torchvision.__version__}")
        return True
    except Exception as exc:
        print(f"[torch/torchvision] FAILED: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download & cache pretrained weights (SBERT + CLIP).")
    parser.add_argument("--check", action="store_true", help="Only report cache status, do not download.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    print("=" * 60)
    print("  DOWNLOAD PRETRAINED WEIGHTS")
    print("=" * 60)
    ok_tv = verify_torchvision()

    if args.check:
        sbert_ok = _cache_ok_sbert()
        clip_ok = _cache_ok_clip()
        print(f"[SBERT] {'cached' if sbert_ok else 'MISSING'}: {SBERT_NAME}")
        print(f"[CLIP]  {'cached' if clip_ok else 'MISSING'}: {CLIP_NAME} ({CLIP_PRETRAINED})")
        print("=" * 60)
        return 0 if (sbert_ok and clip_ok and ok_tv) else 1

    sbert_ok = download_sbert(force=args.force)
    clip_ok = download_clip(force=args.force)
    print("=" * 60)
    if sbert_ok and clip_ok and ok_tv:
        print("  All weights are ready. You can run rebuild_all.py offline now.")
        return 0
    print("  Some weights failed to download. Check Internet connection and retry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
