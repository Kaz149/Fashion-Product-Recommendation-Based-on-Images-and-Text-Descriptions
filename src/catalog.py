from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config import CATEGORIES, RETRIEVAL
from src.filters import FACET_COLUMNS, build_mask, facet_counts, norm_value

PRODUCT_FIELDS = ('id', 'product_name', 'description', 'category', 'color', 'material', 'style', 'image_path')


def _row_to_product(row: Any, owner: Any = None) -> Dict[str, Any]:
    def pick(field: str) -> str:
        try:
            value = row.get(field, '')
        except Exception:
            value = ''
        return '' if value is None else str(value)

    product = {field: pick(field) for field in PRODUCT_FIELDS}
    product['category'] = CATEGORIES.normalize(product['category'])
    if product['image_path'] and owner is not None:
        product['image_path'] = str(owner.resolve_image_path(product['image_path']))
    return product


class Catalog:
    def __init__(self, retriever: Any):
        self.retriever = retriever
        self._df: Optional[pd.DataFrame] = None
        self._id_index: Optional[Dict[str, int]] = None

    # -------------------------------------------------------------- dataframe
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            df = self.retriever.load_metadata().copy()
            df['id'] = df['id'].astype(str)
            for column in ('category', 'color', 'material', 'style'):
                if column in df.columns:
                    df[column] = df[column].map(norm_value)
            df['category'] = df['category'].map(CATEGORIES.normalize)
            self._df = df.reset_index(drop=True)
        return self._df

    @property
    def id_index(self) -> Dict[str, int]:
        if self._id_index is None:
            self._id_index = {str(pid): i for i, pid in enumerate(self.df['id'])}
        return self._id_index

    def __len__(self) -> int:
        return len(self.df)

    def resolve_image_path(self, image_path: str) -> Any:
        resolver = getattr(self.retriever, 'resolve_image_path', None)
        if callable(resolver):
            return resolver(image_path)
        from pathlib import Path
        return Path(image_path)

    # ------------------------------------------------------------- read access
    def get(self, product_id: Any) -> Optional[Dict[str, Any]]:
        index = self.id_index.get(str(product_id).strip())
        if index is None:
            return None
        return _row_to_product(self.df.iloc[index], owner=self)

    def get_many(self, product_ids: Sequence[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pid in product_ids:
            product = self.get(pid)
            if product is not None:
                out.append(product)
        return out

    def products(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        mask = build_mask(self.df, filters)
        subset = self.df if mask is None else self.df[mask]
        rows = subset.to_dict('records')
        if limit is not None:
            rows = rows[: int(limit)]
        return [_row_to_product(row, owner=self) for row in rows]

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        mask = build_mask(self.df, filters)
        return int(len(self.df)) if mask is None else int(mask.sum())

    def facets(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[tuple]]:
        return facet_counts(self.df, filters, FACET_COLUMNS)

    def category_shelf(self, category: str, limit: int = 8) -> List[Dict[str, Any]]:
        bucket = self.df[self.df['category'] == CATEGORIES.normalize(category)].sort_values('id').head(limit)
        return [_row_to_product(row, owner=self) for row in bucket.to_dict('records')]

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: Optional[str] = None,
        image_path: Optional[str] = None,
        mode: str = 'hybrid',
        model_image: str = 'clip',
        model_text: str = RETRIEVAL.DEFAULT_TEXT,
        alpha: float = RETRIEVAL.FUSION_ALPHA_DEFAULT,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = RETRIEVAL.DEFAULT_TOP_K,
        exclude: Sequence[Any] = (),
    ) -> List[Dict[str, Any]]:
        query = (query or '').strip()
        top_k = max(1, min(int(top_k), RETRIEVAL.DEFAULT_MAX_K))
        has_text = bool(query)
        has_image = bool(image_path)
        if not has_text and not has_image:
            return []

        if mode == 'text' or (mode == 'hybrid' and has_text and not has_image):
            raw = self.retriever.search_by_text(query, model=model_text, top_k=top_k, filters=filters)
        elif mode == 'image' or (mode == 'hybrid' and has_image and not has_text):
            raw = self.retriever.search_by_image(image_path, model=model_image, top_k=top_k, filters=filters)
        else:
            raw = self.retriever.search_multimodal(
                img_path=image_path if has_image else None,
                text=query if has_text else None,
                alpha=alpha,
                top_k=top_k,
                filters=filters,
                attribute_boost=True,
            )
        blocked = {str(pid) for pid in exclude}
        return [self._finalize(item) for item in raw if str(item.get('id')) not in blocked]

    def _finalize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        product = dict(item)
        product['id'] = str(product.get('id', ''))
        product['category'] = CATEGORIES.normalize(str(product.get('category', '')))
        for field in ('color', 'material', 'style'):
            product[field] = norm_value(product.get(field, ''))
        product['similarity'] = float(max(0.0, min(1.0, float(product.get('similarity', 0.0)))))
        return product

    # ------------------------------------------------------------ product page
    def similar(
        self,
        product_id: Any,
        top_k: int = 8,
        model: str = 'clip',
        same_category: bool = True,
    ) -> List[Dict[str, Any]]:
        product = self.get(product_id)
        if product is None:
            return []
        pool = max(int(top_k) * 6, int(top_k) + 5)
        try:
            raw = self.retriever.search_by_image(product['image_path'], model=model, top_k=pool)
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            item = self._finalize(item)
            if item['id'] == str(product_id):
                continue
            if same_category and item['category'] != product['category']:
                continue
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def outfit_targets(self, product_id: Any) -> List[str]:
        product = self.get(product_id)
        if product is None:
            return list(CATEGORIES.VALID_FASHION)
        targets = list(CATEGORIES.OUTFIT_MAP.get(product['category'], ()))
        if not targets:
            targets = [c for c in CATEGORIES.VALID_FASHION if c != product['category']]
        return targets

    def complete_the_look(
        self,
        product_id: Any,
        target_category: Optional[str] = None,
        mode: str = 'tone_on_tone',
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        product = self.get(product_id)
        if product is None:
            return []
        targets = self.outfit_targets(product_id)
        target = CATEGORIES.normalize(target_category or (targets[0] if targets else 'shoes'))
        try:
            raw = self.retriever.outfit_match(product_id, target_category=target, top_k=top_k, match_mode=mode)
        except Exception:
            return []
        return [self._finalize(item) for item in raw if str(item.get('id')) != str(product_id)]

    def build_outfit(self, product_id: Any, mode: str = 'tone_on_tone', per_category: int = 1) -> List[Dict[str, Any]]:
        anchor = self.get(product_id)
        if anchor is None:
            return []
        items = [anchor]
        for category in self.outfit_targets(product_id):
            items.extend(self.complete_the_look(product_id, category, mode=mode, top_k=per_category))
        return items
