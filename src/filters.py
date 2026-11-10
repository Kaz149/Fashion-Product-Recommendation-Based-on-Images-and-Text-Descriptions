from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

FACET_COLUMNS: tuple = ('category', 'color', 'material', 'style', 'gender')
FACET_LABELS: Dict[str, str] = {
    'category': 'Danh mục',
    'color': 'Màu sắc',
    'material': 'Chất liệu',
    'style': 'Phong cách',
    'gender': 'Giới tính',
}
HIDDEN_VALUES = {'', 'nan', 'none', 'null', 'other', 'unknown'}


def norm_value(value: Any) -> str:
    return str(value if value is not None else '').strip().lower()


def _as_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        items = list(value)
    except TypeError:
        items = [value]
    return [v for v in (norm_value(i) for i in items) if v]


def active_filters(filters: Optional[Mapping[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for column, value in (filters or {}).items():
        values = _as_values(value)
        if values:
            out[norm_value(column)] = values
    return out


def build_mask(
    df: pd.DataFrame,
    filters: Optional[Mapping[str, Any]],
    columns: Sequence[str] = FACET_COLUMNS,
) -> Optional[np.ndarray]:
    active = active_filters(filters)
    if not active:
        return None
    mask = np.ones(len(df), dtype=bool)
    for column, wanted in active.items():
        if column not in df.columns:
            continue
        mask &= df[column].map(norm_value).isin(wanted).to_numpy()
    return mask

def facet_counts(
    df: pd.DataFrame,
    filters: Optional[Mapping[str, Any]] = None,
    columns: Sequence[str] = FACET_COLUMNS,
) -> Dict[str, List[tuple]]:
    active = active_filters(filters)
    out: Dict[str, List[tuple]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        siblings = {k: v for k, v in active.items() if k != column}
        sibling_mask = build_mask(df, siblings, columns)
        subset = df if sibling_mask is None else df[sibling_mask]
        if len(subset) == 0:
            out[column] = []
            continue
        counts = subset[column].map(norm_value).value_counts()
        out[column] = [
            (str(value), int(count))
            for value, count in counts.items()
            if str(value) not in HIDDEN_VALUES
        ]
    return out


def facet_options(counts: Dict[str, List[tuple]], column: str, selected: Sequence[str] = ()) -> List[str]:
    values = [value for value, _ in counts.get(column, [])]
    for value in selected:
        if value not in values:
            values.append(value)
    return values


def filter_summary(filters: Optional[Mapping[str, Any]]) -> List[str]:
    active = active_filters(filters)
    return [
        f'{FACET_LABELS.get(column, column)}: {value}'
        for column, values in active.items()
        for value in values
    ]
