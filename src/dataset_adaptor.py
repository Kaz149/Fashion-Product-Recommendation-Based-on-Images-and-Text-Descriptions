from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import pandas as pd
from PIL import Image, UnidentifiedImageError
from config import CATEGORIES, PATHS
REQUIRED_COLUMNS = ['id', 'image_path', 'product_name', 'description', 'color', 'material', 'style', 'category']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
_IMAGENET_TO_CATEGORY = {'n02916936': 'shirt', 'n04133789': 'shoes', 'n03047690': 'shoes', 'n02747177': 'bag', 'n04325704': 'watch', 'n03450230': 'dress', 'n02807133': 'underwear'}
_FILENAME_KEYWORDS = {'shirt': ['t-shirt', 'tshirt', 'tee', 'shirt', 'blouse', 'top'], 'trouser': ['trouser', 'pant', 'pants', 'jeans', 'jean', 'short', 'shorts'], 'shoes': ['shoes', 'shoe', 'sneaker', 'sneakers', 'boot', 'boots', 'sandal', 'sandals', 'footwear', 'loafer'], 'bag': ['bag', 'purse', 'backpack', 'handbag', 'tote', 'clutch', 'wallet'], 'watch': ['watch', 'wristwatch', 'clock'], 'dress': ['dress', 'gown', 'skirt', 'outfit'], 'underwear': ['underwear', 'underpant', 'brief', 'bra', 'panties', 'lingerie']}
_VIETNAMESE_CATEGORY = {'shirt': 'Áo', 'trouser': 'Quần', 'shoes': 'Giày', 'bag': 'Túi xách', 'watch': 'Đồng hồ', 'dress': 'Váy đầm', 'underwear': 'Đồ lót'}
_VIETNAMESE_COLOR = {'white': 'trắng', 'black': 'đen', 'blue': 'xanh dương', 'navy': 'xanh đậm', 'gray': 'xám', 'pink': 'hồng', 'red': 'đỏ', 'olive': 'xanh olive', 'brown': 'nâu', 'beige': 'be', 'green': 'xanh lá', 'yellow': 'vàng', 'purple': 'tím'}
_VIETNAMESE_MATERIAL = {'cotton': 'cotton', 'polyester': 'polyester', 'leather': 'da', 'denim': 'jean', 'linen': 'lanh', 'canvas': 'vải bạt', 'silk': 'lụa', 'wool': 'len', 'suede': 'da lộn', 'nylon': 'nylon', 'rubber': 'cao su', 'satin': 'satin'}
_VIETNAMESE_STYLE = {'casual': 'thường nhật', 'formal': 'trang trọng', 'sporty': 'thể thao', 'office': 'văn phòng', 'vintage': 'cổ điển', 'minimalist': 'tối giản', 'streetwear': 'đường phố', 'elegant': 'thanh lịch', 'classic': 'kinh điển', 'modern': 'hiện đại'}
_ENGLISH_ADJECTIVE = ['Classic', 'Premium', 'Stylish', 'Comfortable', 'Modern', 'Elegant', 'Casual', 'Versatile']

def filter_fashion_only(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = set(CATEGORIES.VALID_FASHION)
    kept = []
    for r in rows:
        cat_raw = str(r.get('category', '')).strip()
        cat = CATEGORIES.normalize(cat_raw) if cat_raw else ''
        if cat in valid:
            r['category'] = cat
            kept.append(r)
    return kept

class DatasetLoader:

    def __init__(self, format: Literal['A', 'B', 'C'], dataset_path: str | Path):
        self.format: Literal['A', 'B', 'C'] = format
        self.dataset_path: Path = Path(dataset_path)
        self._resnet_model = None
        self._resnet_transform = None
        self._resnet_classes = None
        self._filename_map: Dict[str, str] = {}
        self._load_filename_map_override()

    def _load_filename_map_override(self) -> None:
        map_path = PATHS.DATA_CUSTOM_FILENAME_MAP
        if map_path.exists():
            try:
                with open(map_path, 'r', encoding='utf-8') as f:
                    self._filename_map = json.load(f)
            except Exception:
                self._filename_map = {}

    def _load_format_a(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        idx = 1
        if not self.dataset_path.exists():
            return rows
        for category_dir in sorted(self.dataset_path.iterdir()):
            if not category_dir.is_dir():
                continue
            raw_category = category_dir.name
            normalized_cat = CATEGORIES.normalize(raw_category)
            for img_file in sorted(category_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({'id': f'prod_{idx:06d}', 'image_path': str(img_file.resolve()), 'product_name': '', 'description': '', 'color': '', 'material': '', 'style': '', 'category': normalized_cat, '_raw_image_path': img_file.resolve()})
                    idx += 1
        return rows

    def _load_format_b(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        csv_candidates = []
        if self.dataset_path == PATHS.DATA_RAW_IMAGES:
            csv_candidates.append(PATHS.DATA_CUSTOM_METADATA)
        else:
            csv_candidates.append(self.dataset_path / 'metadata.csv')
            csv_candidates.append(PATHS.DATA_CUSTOM_METADATA)
        csv_path = None
        for cand in csv_candidates:
            if Path(cand).exists():
                csv_path = Path(cand)
                break
        if csv_path is None:
            return rows
        try:
            df = pd.read_csv(csv_path, dtype=str).fillna('')
        except Exception:
            return rows
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        base_dir = csv_path.parent
        for _, r in df.iterrows():
            img_path = str(r['image_path']).strip()
            if not img_path:
                continue
            p = Path(img_path)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            cat_raw = str(r['category']).strip()
            rows.append({'id': str(r['id']).strip() or '', 'image_path': str(p), 'product_name': str(r['product_name']).strip(), 'description': str(r['description']).strip(), 'color': str(r['color']).strip(), 'material': str(r['material']).strip(), 'style': str(r['style']).strip(), 'category': CATEGORIES.normalize(cat_raw), '_raw_image_path': p})
        for i, row in enumerate(rows, 1):
            if not row['id']:
                row['id'] = f'prod_{i:06d}'
        return rows

    def _init_resnet(self) -> None:
        if self._resnet_model is not None:
            return
        try:
            import torch
            from torchvision import models, transforms
            weights = models.ResNet50_Weights.DEFAULT
            self._resnet_model = models.resnet50(weights=weights)
            self._resnet_model.eval()
            self._resnet_transform = weights.transforms()
            self._resnet_classes = weights.meta['categories']
        except Exception:
            self._resnet_model = None

    def _predict_category_resnet(self, img_path: Path) -> Optional[str]:
        if self._resnet_model is None:
            return None
        try:
            import torch
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                batch = self._resnet_transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = self._resnet_model(batch)
            topk = torch.topk(logits, k=min(3, logits.shape[1]))
            top_indices = topk.indices.squeeze().tolist()
            if not isinstance(top_indices, list):
                top_indices = [top_indices]
            all_cats_lower = []
            for idx in top_indices:
                cat_name = self._resnet_classes[idx] if self._resnet_classes else ''
                all_cats_lower.append(cat_name.lower())
            keyword_map = [(['vest', 'shirt', 'jersey', 'cardigan', 'sweatshirt', 'coat', 'apron', 'kimono', 'pajama', 'brassiere', 'sarong'], 'shirt'), (['jean', 'jeans'], 'trouser'), (['sandal', 'clog', 'shoe', 'boot', 'sneaker', 'running shoe', 'loafer', 'slipper', 'sock'], 'shoes'), (['trunk', 'bag', 'purse', 'backpack', 'wallet', 'backpack', 'handbag', 'tote', 'clutch', 'holster', 'mailbag'], 'bag'), (['watch', 'stopwatch', 'clock', 'stop watch', 'digital watch', 'analog clock', 'wall clock'], 'watch'), (['gown', 'dress', 'overskirt', 'kimono'], 'dress'), (['bathing cap', 'swimsuit', 'bra', 'maillot', 'bikini', 'brassiere'], 'underwear')]
            for cat_lower in all_cats_lower:
                for kws, our_cat in keyword_map:
                    for kw in kws:
                        if kw in cat_lower:
                            return our_cat
            return None
        except Exception:
            return None

    def _predict_category_flatfolder(self, img_path: Path) -> str:
        fname = img_path.name.lower()
        if fname in self._filename_map:
            return CATEGORIES.normalize(self._filename_map[fname])
        stem = img_path.stem.lower()
        if stem in self._filename_map:
            return CATEGORIES.normalize(self._filename_map[stem])
        resnet_cat = self._predict_category_resnet(img_path)
        if resnet_cat is not None:
            return CATEGORIES.normalize(resnet_cat)
        fname_full = f'{stem} {fname}'
        for our_cat, kws in _FILENAME_KEYWORDS.items():
            for kw in kws:
                if kw in fname_full:
                    return CATEGORIES.normalize(our_cat)
        return 'other'

    def _load_format_c(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.dataset_path.exists():
            return rows
        self._init_resnet()
        img_files = sorted([f for f in self.dataset_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
        try:
            from tqdm import tqdm
            _iter = tqdm(img_files, desc='  Predicting categories', unit='img', ncols=90)
        except Exception:
            _iter = img_files
        for idx, img_file in enumerate(_iter, 1):
            pred_cat = self._predict_category_flatfolder(img_file)
            rows.append({'id': f'prod_{idx:06d}', 'image_path': str(img_file.resolve()), 'product_name': '', 'description': '', 'color': '', 'material': '', 'style': '', 'category': pred_cat, '_raw_image_path': img_file.resolve()})
        return rows

    def load_rows(self) -> List[Dict[str, Any]]:
        if self.format == 'A':
            rows = self._load_format_a()
        elif self.format == 'B':
            rows = self._load_format_b()
        elif self.format == 'C':
            rows = self._load_format_c()
        else:
            raise ValueError(f'Unknown format: {self.format}')
        return filter_fashion_only(rows)
_COLOR_PROTOTYPES = {'white': (245, 245, 245), 'black': (22, 22, 22), 'gray': (128, 128, 128), 'red': (200, 30, 30), 'blue': (30, 80, 200), 'navy': (20, 30, 80), 'green': (30, 150, 70), 'yellow': (240, 210, 60), 'pink': (240, 150, 180), 'brown': (120, 70, 30), 'beige': (220, 200, 170), 'olive': (110, 110, 60), 'purple': (130, 60, 180), 'orange': (235, 130, 40)}

def extract_color_from_image(img_path: str | Path | None) -> Optional[str]:
    if not img_path:
        return None
    try:
        p = Path(str(img_path))
        if not p.exists():
            return None
        import numpy as np
        with Image.open(p) as im:
            im = im.convert('RGB').resize((64, 64), Image.LANCZOS)
            arr = __import__('numpy').asarray(im, dtype=float).reshape(-1, 3)
            mask = ~((arr[:, 0] > 225) & (arr[:, 1] > 225) & (arr[:, 2] > 225))
            fg = arr[mask] if mask.sum() > 200 else arr
            c = np.median(fg, axis=0)
            import math
            best, best_d = ('gray', 1e+18)
            for name, proto in _COLOR_PROTOTYPES.items():
                d = sum(((float(x) - float(y)) ** 2 for x, y in zip(c, proto)))
                if d < best_d:
                    best, best_d = (name, d)
            return best
    except Exception:
        return None

class MetadataGenerator:

    def __init__(self, seed: int=42):
        self.global_seed = seed

    def _get_rng(self, seed_key: str) -> random.Random:
        h = hashlib.md5(f'{self.global_seed}|{seed_key}'.encode('utf-8')).hexdigest()
        seed_int = int(h[:8], 16)
        return random.Random(seed_int)

    def _pick_color(self, rng: random.Random) -> str:
        return rng.choice(list(CATEGORIES.COLORS))

    def _pick_material(self, rng: random.Random, category: str) -> str:
        materials = list(CATEGORIES.MATERIALS)
        if category == 'shoes':
            pref = ['leather', 'canvas', 'suede', 'rubber']
        elif category == 'bag':
            pref = ['leather', 'canvas', 'suede', 'nylon']
        elif category == 'dress':
            pref = ['cotton', 'silk', 'satin', 'linen']
        elif category == 'trouser':
            pref = ['denim', 'cotton', 'polyester', 'linen']
        elif category == 'watch':
            pref = ['leather', 'metal']
            materials = [m for m in materials if m in CATEGORIES.MATERIALS]
        elif category == 'underwear':
            pref = ['cotton', 'silk', 'satin', 'nylon']
        else:
            pref = ['cotton', 'polyester', 'linen', 'denim']
        valid_pref = [m for m in pref if m in CATEGORIES.MATERIALS]
        pool = valid_pref + materials
        seen = set()
        uniq_pool = []
        for m in pool:
            if m not in seen:
                seen.add(m)
                uniq_pool.append(m)
        if rng.random() < 0.8 and valid_pref:
            return rng.choice(valid_pref)
        return rng.choice(uniq_pool)

    def _pick_style(self, rng: random.Random, category: str) -> str:
        styles = list(CATEGORIES.STYLES)
        if category in ('shirt', 'trouser'):
            pref = ['casual', 'office', 'formal', 'modern']
        elif category == 'dress':
            pref = ['elegant', 'formal', 'casual', 'vintage']
        elif category == 'shoes':
            pref = ['casual', 'sporty', 'formal', 'streetwear']
        elif category == 'bag':
            pref = ['elegant', 'casual', 'modern', 'classic']
        elif category == 'watch':
            pref = ['classic', 'elegant', 'modern', 'formal']
        else:
            pref = ['casual', 'modern', 'classic']
        valid_pref = [s for s in pref if s in CATEGORIES.STYLES]
        if rng.random() < 0.75 and valid_pref:
            return rng.choice(valid_pref)
        return rng.choice(styles)

    def _build_product_name(self, color: str, material: str, style: str, category: str, rng: random.Random) -> str:
        adj = rng.choice(_ENGLISH_ADJECTIVE)
        cat_label = category
        parts = [color.capitalize(), style, material, cat_label]
        if rng.random() < 0.5:
            parts.insert(0, adj)
        return ' '.join((p for p in parts if p)).capitalize()

    def _build_description(self, color: str, material: str, style: str, category: str, product_name: str, rng: random.Random) -> str:
        templates_eng = [f'{product_name} crafted from high-quality {material}. Perfect for {style} occasions.', f'Made of premium {material} in {color}, this {category} offers {style} comfort and style.', f'A {style} {category} in {color} {material}. Ideal for everyday wear.', f'High-quality {color} {category} made of {material}. {style} design for versatile use.']
        desc_eng = rng.choice(templates_eng)
        vi_cat = _VIETNAMESE_CATEGORY.get(category, 'Sản phẩm')
        vi_color = _VIETNAMESE_COLOR.get(color, color)
        vi_material = _VIETNAMESE_MATERIAL.get(material, material)
        vi_style = _VIETNAMESE_STYLE.get(style, style)
        templates_vie = [f'{vi_cat} {vi_color} chất liệu {vi_material} phong cách {vi_style} cho mọi dịp.', f'Sản phẩm {vi_cat} {vi_style} màu {vi_color} làm từ {vi_material} cao cấp.', f'{vi_cat} {vi_color} pha trộn {vi_material} - thiết kế {vi_style} thời thượng.']
        desc_vie = rng.choice(templates_vie)
        return f'{desc_eng} - {desc_vie}'

    def fill_defaults(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        seed_key = out.get('id') or out.get('_raw_image_path') or str(random.random())
        rng = self._get_rng(str(seed_key))
        cat_raw = str(out.get('category', '')).strip()
        if cat_raw:
            category = CATEGORIES.normalize(cat_raw)
        else:
            category = 'other'
        if category == 'other':
            pass
        color = str(out.get('color', '')).strip().lower()
        if color not in CATEGORIES.COLORS:
            scanned = extract_color_from_image(out.get('image_path') or out.get('_raw_image_path'))
            color = scanned if scanned else self._pick_color(rng)
        material = str(out.get('material', '')).strip().lower()
        if material not in CATEGORIES.MATERIALS:
            material = self._pick_material(rng, category)
        style = str(out.get('style', '')).strip().lower()
        if style not in CATEGORIES.STYLES:
            style = self._pick_style(rng, category)
        product_name = str(out.get('product_name', '')).strip()
        if not product_name:
            product_name = self._build_product_name(color, material, style, category, rng)
        description = str(out.get('description', '')).strip()
        if not description or ' - ' not in description:
            description = self._build_description(color, material, style, category, product_name, rng)
        out['category'] = category
        out['color'] = color
        out['material'] = material
        out['style'] = style
        out['product_name'] = product_name
        out['description'] = description
        return out

    @staticmethod
    def build_corpus_text(row: Dict[str, Any]) -> str:
        pn = str(row.get('product_name', '')).strip()
        ds = str(row.get('description', '')).strip()
        cl = str(row.get('color', '')).strip()
        mt = str(row.get('material', '')).strip()
        st = str(row.get('style', '')).strip()
        attr = ' '.join((x for x in [cl, mt, st] if x)).strip()
        return f'{pn} [SEP] {ds} [SEP] {attr}'

    @staticmethod
    def build_corpus_list(rows: List[Dict[str, Any]]) -> List[str]:
        return [MetadataGenerator.build_corpus_text(r) for r in rows]

def sanitize_images(raw_paths_with_ids: List[Tuple[Any, Path]], dst_dir: Path, size: int=224) -> Tuple[List[Tuple[Any, Path]], List[Any]]:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    success: List[Tuple[Any, Path]] = []
    failed: List[Any] = []
    for item_id, raw_path in raw_paths_with_ids:
        try:
            p = Path(raw_path)
            if not p.exists() or not p.is_file():
                failed.append(item_id)
                continue
            if p.stat().st_size == 0:
                failed.append(item_id)
                continue
            with Image.open(p) as img:
                img = img.convert('RGB')
                img = img.resize((size, size), Image.LANCZOS)
                dst_path = dst_dir / f'{item_id}.jpg'
                img.save(dst_path, format='JPEG', quality=90, optimize=True)
                if dst_path.stat().st_size == 0:
                    failed.append(item_id)
                    continue
            success.append((item_id, dst_path.resolve()))
        except (UnidentifiedImageError, OSError, ValueError):
            failed.append(item_id)
        except Exception:
            failed.append(item_id)
    return (success, failed)

def save_metadata(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    out_cols = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS and (not c.startswith('_'))]
    df[out_cols].to_csv(csv_path, index=False, encoding='utf-8')

def compute_dataset_fingerprint(metadata_csv_path: Path, processed_images_dir: Path) -> str:
    hasher = hashlib.sha256()
    meta = Path(metadata_csv_path)
    if meta.exists():
        with open(meta, 'rb') as f:
            hasher.update(hashlib.sha256(f.read()).digest())
    img_dir = Path(processed_images_dir)
    if img_dir.exists():
        files = sorted([p.name for p in img_dir.iterdir() if p.is_file()])
        hasher.update('\n'.join(files).encode('utf-8'))
    return hasher.hexdigest()

def is_embedding_stale(manifest_path: Path=PATHS.EMB_MANIFEST, metadata_csv_path: Path=PATHS.DATA_PROCESSED_METADATA, processed_images_dir: Path=PATHS.DATA_PROCESSED_IMAGES) -> bool:
    manifest = Path(manifest_path)
    if not manifest.exists():
        return True
    try:
        with open(manifest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        stored_fp = str(data.get('dataset_fingerprint', '')).strip()
        if not stored_fp:
            return True
        current_fp = compute_dataset_fingerprint(metadata_csv_path, processed_images_dir)
        return stored_fp != current_fp
    except Exception:
        return True
__all__ = ['DatasetLoader', 'MetadataGenerator', 'sanitize_images', 'save_metadata', 'compute_dataset_fingerprint', 'is_embedding_stale', 'filter_fashion_only', 'REQUIRED_COLUMNS', 'IMAGE_EXTENSIONS']
