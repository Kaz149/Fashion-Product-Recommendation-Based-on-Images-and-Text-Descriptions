import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import train_test_split as _sklearn_split
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

from packages.preprocess.text_preprocess import (
    map_category_vi_to_en,
    map_color_vi_to_en,
    map_style_vi_to_en,
    map_material_vi_to_en,
    clean_text,
)


def _np_train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=None,
):

    rng = np.random.RandomState(random_state)
    n = len(indices)
    idx_arr = np.asarray(list(indices))
    # Xác định số lượng phần tử tập test (float = tỷ lệ %, int = số phần tử cứng)
    n_test = int(round(n * test_size)) if isinstance(test_size, float) else int(test_size)
    n_test = max(0, min(n_test, n - 1)) if n > 1 else 0
    if stratify is None:
        # Không stratified -> chỉ permutation ngẫu nhiên đơn giản
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        return idx_arr[train_idx].tolist(), idx_arr[test_idx].tolist()
    stratify_arr = np.asarray(list(stratify))
    # Gom nhóm theo NHÃN (vd category = shirt/ pants/ shoes riêng từng bucket)
    groups = {}
    for i, g in enumerate(stratify_arr):
        groups.setdefault(str(g), []).append(i)
    train_idx, test_idx = [], []
    for g, members in groups.items():
        members = np.asarray(members)
        n_g = len(members)
        # Lấy TEST_SIZE TỪNG NHÓM RIÊNG -> tỷ lệ shirt trong tập train = trong tập test
        g_test = max(0, min(int(round(n_g * test_size)), n_g - 1)) if n_g > 1 else 0
        perm = rng.permutation(n_g)
        test_idx.extend(members[perm[:g_test]].tolist())
        train_idx.extend(members[perm[g_test:]].tolist())
    # Xáo trộn kết quả cuối (vì group theo thứ tự list keys sẽ gom bucket)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return idx_arr[train_idx].tolist(), idx_arr[test_idx].tolist()


def _train_test_split(*args, **kwargs):
    if _HAS_SKLEARN:
        return _sklearn_split(*args, **kwargs)
    return _np_train_test_split(*args, **kwargs)


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(schema_path: str = "pipeline/schemas/product_schema.json") -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonicalize_vi_fields(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    def _fill_from_vi(en_col: str, vi_col: str, mapper_fn, vi_only_default=None):
        has_en = en_col in df.columns
        has_vi = vi_col in df.columns
        # Trường hợp CSV không có cả EN lẫn vi, tạo cột EN default 
        if not has_en and not has_vi:
            if vi_only_default is not None:
                df[en_col] = vi_only_default
            return df
        if has_en:
            series = df[en_col].where(df[en_col].notna() & (df[en_col].astype(str).str.strip() != ""))
            if has_vi:
                series = series.fillna(df[vi_col])
        else:
            # Không có cột EN thì dùng toàn bộ vi làm nguồn
            series = df[vi_col]
        if mapper_fn is not None:
            df[en_col] = series.astype(str).map(
                lambda v: mapper_fn(v) if isinstance(v, str) and v.strip() else (vi_only_default if vi_only_default is not None else "")
            )
        else:
            df[en_col] = series.fillna("").astype(str)
        return df

    # 4 trường dữ liệu (category, color, style, material) + tên + mô tả
    df = _fill_from_vi("category", "category_vi", map_category_vi_to_en, vi_only_default="other")
    df = _fill_from_vi("color", "color_vi", map_color_vi_to_en, vi_only_default="")
    df = _fill_from_vi("style", "style_vi", map_style_vi_to_en, vi_only_default="casual")
    df = _fill_from_vi("material", "material_vi", map_material_vi_to_en, vi_only_default="")
    df = _fill_from_vi("name", "name_vi", None, vi_only_default="")
    df = _fill_from_vi("description", "description_vi", None, vi_only_default="")

    for col in ["subcategory", "gender", "brand", "name_vi", "description_vi",
                "category_vi", "color_vi", "style_vi", "material_vi"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    valid_categories = {"shirt", "pants", "shoes", "dress", "jacket", "skirt", "accessory", "other"}
    df.loc[~df["category"].isin(valid_categories), "category"] = "other"

    return df


def load_metadata(
    csv_path: str,
    validate_image_paths: bool = False,
    image_root: Optional[str] = None,
    canonicalize_vi: bool = True,
) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    has_any_name = ("name" in df.columns) or ("name_vi" in df.columns)
    has_any_category = ("category" in df.columns) or ("category_vi" in df.columns)
    required_native = {"id", "image_path"}
    if not has_any_name:
        required_native.add("name")
    if not has_any_category:
        required_native.add("category")
    missing = required_native - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns (or _vi variants): {sorted(missing)}")

    if canonicalize_vi:
        df = _canonicalize_vi_fields(df)

    if validate_image_paths:
        root = Path(image_root) if image_root else Path(csv_path).parent
        df["_image_exists"] = df["image_path"].apply(
            lambda p: os.path.isfile(root / p) or os.path.isfile(p)
        )
        before = len(df)
        df = df[df["_image_exists"]].drop(columns=["_image_exists"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"[dataset_loader] Dropped {dropped} rows with missing images")
    for col in ["color", "material", "style", "description", "subcategory", "gender", "brand"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["category"] = df["category"].fillna("other").astype(str)
    return df


def assign_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by: Optional[str] = "category",
) -> pd.DataFrame:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    df = df.copy().reset_index(drop=True)
    n = len(df)
    stratify = df[stratify_by] if stratify_by and stratify_by in df.columns and n >= 2 else None
    if n == 0:
        df["split"] = pd.Series([], dtype="object")
        return df
    test_size = test_ratio
    train_val_idx, test_idx = _train_test_split(
        df.index.tolist(),
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify if stratify is not None and len(stratify.unique()) <= n // 2 else None,
    )
    val_relative = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0
    df_tv = df.loc[train_val_idx].reset_index(drop=True)
    stratify_tv = df_tv[stratify_by] if stratify_by and stratify_by in df_tv.columns and len(df_tv) >= 2 else None
    train_idx, val_idx = _train_test_split(
        df_tv.index.tolist(),
        test_size=val_relative,
        random_state=random_seed,
        stratify=stratify_tv if stratify_tv is not None and len(stratify_tv.unique()) <= len(df_tv) // 2 else None,
    )
    df.loc[df_tv.index[train_idx], "split"] = "train"
    df.loc[df_tv.index[val_idx], "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def load_dataset(
    csv_path: Optional[str] = None,
    config_path: str = "pipeline/config/models.yaml",
    validate_images: bool = False,
    image_root: Optional[str] = None,
    canonicalize_vi: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    if csv_path is None:
        csv_path = cfg["paths"]["metadata_csv"]
    ds = cfg["dataset"]
    df = load_metadata(csv_path, validate_image_paths=validate_images, image_root=image_root, canonicalize_vi=canonicalize_vi)
    df = assign_splits(
        df,
        train_ratio=ds["train_ratio"],
        val_ratio=ds["val_ratio"],
        test_ratio=ds["test_ratio"],
        random_seed=ds["random_seed"],
    )
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def load_all_products(
    csv_path: Optional[str] = None,
    config_path: str = "pipeline/config/models.yaml",
    validate_images: bool = False,
    image_root: Optional[str] = None,
    add_splits: bool = True,
    canonicalize_vi: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if csv_path is None:
        csv_path = cfg["paths"]["metadata_csv"]
    ds = cfg["dataset"]
    df = load_metadata(csv_path, validate_image_paths=validate_images, image_root=image_root, canonicalize_vi=canonicalize_vi)
    if add_splits:
        df = assign_splits(
            df,
            train_ratio=ds["train_ratio"],
            val_ratio=ds["val_ratio"],
            test_ratio=ds["test_ratio"],
            random_seed=ds["random_seed"],
        )
    return df
