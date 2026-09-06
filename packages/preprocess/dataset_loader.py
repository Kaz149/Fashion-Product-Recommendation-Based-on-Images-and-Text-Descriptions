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


def _np_train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=None,
):
    rng = np.random.RandomState(random_state)
    n = len(indices)
    idx_arr = np.asarray(list(indices))
    n_test = int(round(n * test_size)) if isinstance(test_size, float) else int(test_size)
    n_test = max(0, min(n_test, n - 1)) if n > 1 else 0
    if stratify is None:
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        return idx_arr[train_idx].tolist(), idx_arr[test_idx].tolist()
    stratify_arr = np.asarray(list(stratify))
    groups = {}
    for i, g in enumerate(stratify_arr):
        groups.setdefault(str(g), []).append(i)
    train_idx, test_idx = [], []
    for g, members in groups.items():
        members = np.asarray(members)
        n_g = len(members)
        g_test = max(0, min(int(round(n_g * test_size)), n_g - 1)) if n_g > 1 else 0
        perm = rng.permutation(n_g)
        test_idx.extend(members[perm[:g_test]].tolist())
        train_idx.extend(members[perm[g_test:]].tolist())
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


def load_metadata(
    csv_path: str,
    validate_image_paths: bool = False,
    image_root: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"id", "name", "category", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    if csv_path is None:
        csv_path = cfg["paths"]["metadata_csv"]
    ds = cfg["dataset"]
    df = load_metadata(csv_path, validate_image_paths=validate_images, image_root=image_root)
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
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if csv_path is None:
        csv_path = cfg["paths"]["metadata_csv"]
    ds = cfg["dataset"]
    df = load_metadata(csv_path, validate_image_paths=validate_images, image_root=image_root)
    if add_splits:
        df = assign_splits(
            df,
            train_ratio=ds["train_ratio"],
            val_ratio=ds["val_ratio"],
            test_ratio=ds["test_ratio"],
            random_seed=ds["random_seed"],
        )
    return df
