import hashlib, pickle, warnings
from pathlib import Path, PureWindowsPath
from typing import Iterable, Literal, Any, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from config import PATHS, TEXT_MODELS, RETRIEVAL, CATEGORIES
from src.dataset_adaptor import is_embedding_stale
from src.image_embedder import get_image_embedder
from src.text_embedder import get_text_embedder, TFIDFEmbedder
_VN2EN = {'ao thun': 't-shirt shirt', 'áo thun': 't-shirt shirt', 'ao so mi': 'shirt', 'áo sơ mi': 'shirt', 'quan jean': 'jeans trouser denim', 'quần jean': 'jeans trouser denim', 'quan': 'trouser pants', 'quần': 'trouser pants', 'vay': 'dress skirt', 'váy': 'dress skirt', 'dam': 'dress', 'đầm': 'dress', 'giay': 'shoes sneaker', 'giày': 'shoes sneaker', 'tui xach': 'handbag bag', 'túi xách': 'handbag bag', 'den': 'black', 'đen': 'black', 'trang': 'white', 'trắng': 'white', 'do': 'red', 'đỏ': 'red', 'xanh la': 'green', 'xanh lá': 'green', 'xanh duong': 'blue', 'xanh dương': 'blue', 'xanh navy': 'navy', 'xanh đen': 'navy', 'navy': 'navy', 'xanh': 'blue', 'vang': 'yellow', 'vàng': 'yellow', 'nau': 'brown', 'nâu': 'brown', 'hong': 'pink', 'hồng': 'pink', 'cam': 'orange', 'tim': 'purple', 'tím': 'purple', 'xam': 'gray', 'xám': 'gray', 'ghi': 'gray', 'oliue': 'olive', 'olive': 'olive', 'o liu': 'olive', 'xanh rêu': 'olive', 'xanh reu': 'olive', 'be': 'beige', 'kem': 'beige', 'cotton': 'cotton', 'jean': 'denim jeans', 'da': 'leather', 'nam': 'men', 'nu': 'women', 'nữ': 'women'}

def _strip_accents(s: str) -> str:
    import unicodedata
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
_COLORS_EN = ('black', 'white', 'red', 'blue', 'navy', 'green', 'yellow', 'pink', 'brown', 'beige', 'gray', 'grey', 'olive', 'purple', 'orange', 'cream', 'khaki', 'maroon', 'teal', 'cyan', 'mustard', 'coral', 'lavender', 'silver', 'charcoal', 'gold', 'mint')
_STYLES_EN = ('casual', 'formal', 'sporty', 'office', 'vintage', 'minimalist', 'streetwear', 'elegant', 'classic', 'modern')
_MATERIALS_EN = ('cotton', 'polyester', 'leather', 'denim', 'linen', 'canvas', 'silk', 'wool', 'suede', 'nylon', 'rubber', 'satin')
_COLOR_SIMILAR: dict[str, tuple[str, ...]] = {
    'white': ('silver', 'cream', 'beige', 'ivory', 'off-white'),
    'silver': ('white', 'gray', 'cream', 'charcoal'),
    'gray': ('silver', 'charcoal', 'greige'),
    'charcoal': ('black', 'gray', 'silver'),
    'black': ('charcoal',),
    'cream': ('white', 'beige', 'ivory', 'khaki'),
    'beige': ('cream', 'khaki', 'white', 'tan'),
    'khaki': ('beige', 'olive', 'cream'),
    'red': ('maroon', 'crimson', 'coral', 'burgundy'),
    'maroon': ('red', 'crimson', 'burgundy', 'brown'),
    'coral': ('red', 'pink', 'salmon', 'peach'),
    'pink': ('coral', 'lavender', 'salmon'),
    'orange': ('coral', 'mustard', 'gold', 'rust'),
    'mustard': ('gold', 'yellow', 'orange'),
    'gold': ('mustard', 'yellow', 'bronze'),
    'yellow': ('mustard', 'gold', 'cream'),
    'green': ('olive', 'mint', 'teal', 'sage'),
    'mint': ('green', 'teal', 'sage'),
    'olive': ('green', 'khaki', 'brown'),
    'teal': ('cyan', 'green', 'blue', 'turquoise'),
    'cyan': ('teal', 'blue', 'turquoise'),
    'blue': ('navy', 'cyan', 'teal', 'denim'),
    'navy': ('blue', 'black', 'denim'),
    'purple': ('lavender', 'plum', 'violet'),
    'lavender': ('purple', 'pink', 'lilac'),
    'brown': ('maroon', 'olive', 'tan', 'bronze'),
}

def _autocorrect_query(text: str) -> tuple[str, list[str]]:
    import difflib
    vocab: dict[str, str] = {}
    for w in list(_COLORS_EN) + list(_STYLES_EN) + list(_MATERIALS_EN):
        vocab[w] = w
    for alias, canon in {'grey': 'gray', 'tshirt': 'shirt', 't-shirt': 'shirt', 'tee': 'shirt', 'pant': 'trouser', 'pants': 'trouser', 'jeans': 'trouser', 'sneaker': 'shoes', 'sneakers': 'shoes', 'handbag': 'bag', 'gown': 'dress', 'skirt': 'dress'}.items():
        vocab[alias] = canon
    for cat in ('shirt', 'trouser', 'dress', 'shoes', 'bag', 'watch', 'underwear'):
        vocab[cat] = cat
    vocab['ao'] = '\u00e1o'
    # Vietnamese support: map every VN phrase (accented + unaccented) to its
    # canonical EN token (first word of the _VN2EN expansion, e.g. 'ao so mi'->'shirt').
    for vn, en in _VN2EN.items():
        canon = en.split()[0]
        vocab[vn] = canon
        un = _strip_accents(vn)
        if un not in vocab:
            vocab[un] = canon
    fixed: list[str] = []
    # Pass 1: multi-word VN phrases (longest first) so 'xanh duong' is fixed as
    # a unit instead of two separate tokens.
    work = f' {text} '
    for phrase in sorted([k for k in vocab if ' ' in k], key=len, reverse=True):
        import re as _re
        pat = _re.compile(r'(?i)(?<![\w\u00c0-\u1ef9])' + _re.escape(phrase) + r'(?![\w\u00c0-\u1ef9])')
        if pat.search(work):
            work = pat.sub(f' {vocab[phrase]} ', work)
    # Pass 2: token-level fuzzy match (typos like 'trang'/'while'/'cototn').
    out: list[str] = []
    for tok in work.split():
        low = tok.lower().strip('.,!?;:')
        if not low or len(low) < 2:
            out.append(tok)
            continue
        if len(low) == 2 and low not in ('ao',):
            out.append(tok)
            continue
        if low in vocab and vocab[low] != low:
            if low not in ('ao',) or True:
                fixed.append(f'{tok}->{vocab[low]}')
            out.append(vocab[low])
            continue
        best = difflib.get_close_matches(low, vocab.keys(), n=1, cutoff=0.8)
        un_low = _strip_accents(low)
        if not best and un_low != low:
            best = difflib.get_close_matches(un_low, vocab.keys(), n=1, cutoff=0.8)
        if best:
            canon = vocab[best[0]]
            if canon != low:
                fixed.append(f'{tok}->{canon}')
            out.append(canon if tok == low else canon.capitalize() if tok[:1].isupper() else canon)
        else:
            out.append(tok)
    # Display: if the query is Vietnamese, render corrections back in proper
    # Vietnamese (with diacritics) instead of English tokens.
    _EN2VN_DISPLAY = {'white': 'tr\u1eafng', 'black': '\u0111en', 'red': '\u0111\u1ecf',
        'blue': 'xanh d\u01b0\u01a1ng', 'navy': 'xanh navy', 'green': 'xanh l\u00e1',
        'yellow': 'v\u00e0ng', 'pink': 'h\u1ed3ng', 'brown': 'n\u00e2u', 'beige': 'kem',
        'gray': 'x\u00e1m', 'olive': '\u00f4 liu', 'purple': 't\u00edm', 'orange': 'cam',
        'cream': 'kem', 'khaki': 'kaki', 'maroon': '\u0111\u1ecf \u0111\u1eadm',
        'teal': 'xanh ng\u1ecdc', 'cyan': 'xanh cyan', 'mustard': 'v\u00e0ng m\u00f9 t\u1ea1t',
        'coral': 'san h\u00f4', 'lavender': 't\u00edm nh\u1ea1t', 'silver': 'b\u1ea1c',
        'charcoal': 'x\u00e1m \u0111en', 'gold': 'v\u00e0ng \u0111\u1ed3ng', 'mint': 'xanh b\u1ea1c h\u00e0',
        'shirt': '\u00e1o s\u01a1 mi', 'trouser': 'qu\u1ea7n', 'dress': 'v\u00e1y',
        'shoes': 'gi\u00e0y', 'bag': 't\u00fai x\u00e1ch', 'watch': '\u0111\u1ed3ng h\u1ed3',
        'underwear': '\u0111\u1ed3 l\u00f3t', 'cotton': 'cotton', 'polyester': 'polyester',
        'leather': 'da', 'denim': 'jean', 'linen': 'linen', 'canvas': 'canvas',
        'silk': 'l\u1ee5a', 'wool': 'len', 'suede': 'da l\u1ed9n', 'nylon': 'nylon',
        'rubber': 'cao su', 'satin': 'satin', 'casual': 'd\u1ea1o ph\u1ed1',
        'formal': 'c\u00f4ng s\u1edf', 'sporty': 'th\u1ec3 thao', 'office': 'v\u0103n ph\u00f2ng',
        'vintage': 'vintage', 'minimalist': 't\u1ed1i gi\u1ea3n', 'streetwear': 'streetwear',
        'elegant': 'thanh l\u1ecbch', 'classic': 'c\u1ed5 \u0111i\u1ec3n', 'modern': 'hi\u1ec7n \u0111\u1ea1i',
        'ao': '\u00e1o', 'men': 'nam', 'women': 'n\u1eef', 'kids': 'tr\u1ebb em'}
    _VN_PHRASE_FIX = {'cong so': 'c\u00f4ng s\u1edf', 'van phong': 'v\u0103n ph\u00f2ng',
        'the thao': 'th\u1ec3 thao', 'du tiec': 'd\u1ef1 ti\u1ec7c', 'dao pho': 'd\u1ea1o ph\u1ed1'}
    corrected_en = ' '.join(out)
    probe = f' {text.lower()} '
    is_vn = any(vn in _strip_accents(probe) for vn in _VN_PHRASE_FIX) or any(
        _strip_accents(vn) in _strip_accents(probe) for vn in _VN2EN) or any(
        ord(c) > 127 for c in text)
    if is_vn:
        disp = [ _EN2VN_DISPLAY.get(w.lower(), w) for w in corrected_en.split()]
        s = ' '.join(disp)
        for k, v in _VN_PHRASE_FIX.items():
            import re as _re2
            s = _re2.sub(r'(?i)\b' + _re2.escape(k) + r'\b', v, s)
            s = _re2.sub(r'(?i)\b' + _re2.escape(_strip_accents(v)) + r'\b', v, s)
        return (s, fixed)
    return (' '.join(out), fixed)


_VN_COLOR_FIRST = ('xanh lá', 'xanh la', 'xanh dương', 'xanh duong', 'xanh navy', 'xanh đen', 'xanh den', 'xanh rêu', 'xanh reu', 'o liu')
_CAT_WORDS = {'shirt': ('thun', 'sơ mi', 'so mi', 'somi', 'áo', 'ao', 'shirt', 't-shirt', 'tee', 'blouse'), 'trouser': ('quần', 'quan', 'jean', 'trouser', 'pants', 'pant', 'denim'), 'dress': ('váy', 'vay', 'đầm', 'dam', 'dress', 'skirt', 'gown'), 'shoes': ('giày', 'giay', 'shoe', 'shoes', 'sneaker', 'boot', 'sandal'), 'bag': ('túi', 'tui', 'xách', 'xach', 'bag', 'handbag', 'backpack', 'tote'), 'watch': ('đồng hồ', 'dong ho', 'watch'), 'underwear': ('đồ lót', 'do lot', 'underwear', 'lingerie', 'bra')}
_GENDER_WORDS = {'men': ('nam', 'men', 'male', 'man', 'ông'), 'women': ('nữ', 'nu', 'women', 'female', 'woman', 'bà'), 'kids': ('trẻ em', 'tre em', 'kid', 'kids', 'child')}

def _expand_vn_query(text: str) -> str:
    low = f' {text.lower()} '
    low_un = f' {_strip_accents(text.lower())} '
    extra = [en for vn, en in _VN2EN.items() if vn in low or _strip_accents(vn) in low_un]
    return text + ' ' + ' '.join(extra) if extra else text

def parse_query_attrs(text: str):
    low = f' {text.lower()} '
    low_un = f' {_strip_accents(text.lower())} '
    color = next((c for c in _COLORS_EN if f' {c} ' in low), None)
    if color == 'grey':
        color = 'gray'
    if color is None:
        for vn, en in _VN2EN.items():
            if en not in _COLORS_EN + ('gray',):
                continue
            if vn in _VN_COLOR_FIRST and vn in low:
                color = en
                break
        if color is None:
            for vn, en in _VN2EN.items():
                if en not in _COLORS_EN + ('gray',):
                    continue
                if vn in low or _strip_accents(vn) in low_un:
                    color = en
                    break
    cat = next((c for c, words in _CAT_WORDS.items() if any((w in low for w in words))), None)
    style = next((s for s in _STYLES_EN if f' {s} ' in low), None)
    material = next((m for m in _MATERIALS_EN if f' {m} ' in low), None)
    gender = next((g for g, words in _GENDER_WORDS.items() if any((f' {w} ' in low or low.strip() == w for w in words))), None)
    return (color, cat, gender, style, material)

def _resolve_catalog_image_path(image_path: str | Path) -> Path:
    stored_path = str(image_path).strip()
    # PureWindowsPath đọc được cả đường dẫn Windows lẫn tên dùng dấu '/'.
    filename = PureWindowsPath(stored_path).name
    local_path = PATHS.DATA_PROCESSED_IMAGES / filename
    if filename and local_path.is_file():
        return local_path.resolve()
    original_path = Path(stored_path)
    if not original_path.is_absolute():
        original_path = PATHS.PROJECT_ROOT / original_path
    return original_path

class Retriever:

    def __init__(self, lazy=True, verbose=False):
        self.verbose = verbose
        self.metadata: Optional[pd.DataFrame] = None
        self.product_ids: Optional[List[Any]] = None
        self.image_paths: Optional[List[Path]] = None
        self._stored_image_paths: List[str] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self._img_embedders_cache: Dict = {}
        self._txt_embedders_cache: Dict = {}
        self._nn_index_cache: Dict[str, NearestNeighbors] = {}
        if not lazy:
            self.load_all()

    def load_all(self) -> None:
        self.load_metadata()
        self.load_index_mapping()
        for name in ['resnet50', 'efficientnet_b0', 'clip_image', 'tfidf', 'sbert', 'clip_text']:
            try:
                self.get_embedding(name)
            except FileNotFoundError:
                pass

    def load_metadata(self) -> pd.DataFrame:
        if self.metadata is None:
            self.metadata = pd.read_csv(PATHS.DATA_PROCESSED_METADATA).reset_index(drop=True)
            self.metadata['image_path'] = self.metadata['image_path'].map(
                lambda path: str(_resolve_catalog_image_path(path))
            )
        return self.metadata

    def load_index_mapping(self):
        if self.product_ids is None:
            with open(PATHS.EMB_INDEX_MAPPING, 'rb') as f:
                m = pickle.load(f)
            self.product_ids = m['product_ids']
            self._stored_image_paths = [str(p) for p in m['image_paths']]
            self.image_paths = [_resolve_catalog_image_path(p) for p in m['image_paths']]
        return (self.product_ids, self.image_paths)

    @staticmethod
    def resolve_image_path(image_path: str | Path) -> Path:
        return _resolve_catalog_image_path(image_path)

    def _lazy_warn_stale(self):
        if is_embedding_stale(PATHS.EMB_MANIFEST, PATHS.DATA_PROCESSED_METADATA, PATHS.DATA_PROCESSED_IMAGES):
            if self.verbose:
                warnings.warn('Embeddings stale - dataset changed. Run rebuild_all.py', UserWarning)

    def get_embedding(self, name: Literal['resnet50', 'efficientnet_b0', 'clip_image', 'tfidf', 'sbert', 'clip_text']) -> np.ndarray:
        if name not in self.embeddings:
            attr_map = {'resnet50': 'EMB_RESNET50', 'efficientnet_b0': 'EMB_EFFNET_B0', 'clip_image': 'EMB_CLIP_IMAGE', 'tfidf': 'EMB_TFIDF', 'sbert': 'EMB_SBERT', 'clip_text': 'EMB_CLIP_TEXT'}
            path = getattr(PATHS, attr_map[name])
            if not path.exists():
                raise FileNotFoundError(f'{name} embeddings not found: {path}')
            if name != 'tfidf':
                self.embeddings[name] = np.load(path, mmap_mode='r').astype(np.float32)
            else:
                self.embeddings[name] = np.load(path).astype(np.float32)
            self._lazy_warn_stale()
        return self.embeddings[name]

    def _get_or_fit_nn(self, emb_name: str, emb_corpus: np.ndarray) -> NearestNeighbors:
        if emb_name not in self._nn_index_cache:
            nn = NearestNeighbors(n_neighbors=min(RETRIEVAL.DEFAULT_MAX_K, len(emb_corpus)), metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1)
            nn.fit(emb_corpus)
            self._nn_index_cache[emb_name] = nn
        return self._nn_index_cache[emb_name]

    def _kneighbors(self, emb_name: str, corpus: np.ndarray, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(query_vec, dtype=np.float32))
        topk = max(1, min(int(top_k), RETRIEVAL.DEFAULT_MAX_K))
        nn = self._get_or_fit_nn(emb_name, corpus)
        if topk > nn.n_neighbors:
            nn2 = NearestNeighbors(n_neighbors=topk, metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1).fit(corpus)
            self._nn_index_cache[emb_name] = nn2
            nn = nn2
        dists, idxs = nn.kneighbors(q, n_neighbors=topk)
        return (dists[0], idxs[0])

    def filter_mask(self, filters: Optional[Dict[str, List[str]]]) -> Optional[np.ndarray]:
        if not filters:
            return None
        from src.filters import build_mask
        return build_mask(self.load_metadata(), filters)

    def _search_in_mask(self, emb_name: str, corpus: np.ndarray, query_vec: np.ndarray, top_k: int, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None:
            return self._kneighbors(emb_name, corpus, query_vec, top_k)
        subset = np.flatnonzero(mask)
        if subset.size == 0:
            return (np.empty(0, dtype=int), np.empty(0, dtype=np.float32))
        builder = hashlib.blake2b(np.ascontiguousarray(mask).tobytes(), digest_size=8).hexdigest()
        key = f'{emb_name}::filtered::{builder}'
        k = max(1, min(int(top_k), int(subset.size)))
        nn = self._nn_index_cache.get(key)
        if nn is None or nn.n_neighbors < k:
            nn = NearestNeighbors(n_neighbors=k, metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1).fit(corpus[subset])
            self._nn_index_cache[key] = nn
        dists, local = nn.kneighbors(np.atleast_2d(np.asarray(query_vec, dtype=np.float32)), n_neighbors=k)
        return (dists[0], subset[local[0]])

    def _build_result_row(self, idx: int, sim: float) -> Dict[str, Any]:
        md = self.load_metadata().iloc[idx]
        return {'index': int(idx), 'id': self.product_ids[idx] if self.product_ids else int(md.get('id', idx)), 'image_path': str(self.image_paths[idx]) if self.image_paths else str(md.image_path), 'product_name': str(md.get('product_name', '')), 'description': str(md.get('description', '')), 'category': str(md.get('category', 'other')), 'color': str(md.get('color', '')), 'material': str(md.get('material', '')), 'style': str(md.get('style', '')), 'similarity': float(max(0.0, min(1.0, sim)))}

    def _path_to_index(self) -> Dict[str, int]:
        if getattr(self, '_path_to_index_cache', None) is None:
            mapping: Dict[str, int] = {}
            stored = getattr(self, '_stored_image_paths', None) or []
            for i, ip in enumerate(self.image_paths or []):
                # Index the path as stored in index_mapping and the one resolved
                # for this checkout, so callers using either form hit the map.
                if i < len(stored):
                    mapping[stored[i]] = i
                mapping[str(ip)] = i
                try:
                    mapping[str(Path(ip).resolve())] = i
                except Exception:
                    pass
            self._path_to_index_cache = mapping
        return self._path_to_index_cache

    def _lookup_image_embedding_in_corpus(self, img_path, emb_name: str) -> Optional[np.ndarray]:
        if self.image_paths is None:
            return None
        keys = [str(img_path)]
        try:
            keys.append(str(Path(img_path).resolve()))
        except Exception:
            pass
        mapping = self._path_to_index()
        for key in keys:
            index = mapping.get(key)
            if index is not None:
                emb_all = self.get_embedding(emb_name)
                return np.asarray(emb_all[index], dtype=np.float32).copy()
        return None

    def search_by_image(self, img_path: str | Path, model: Literal['resnet', 'effnet', 'clip']=RETRIEVAL.DEFAULT_IMAGE, top_k: int=RETRIEVAL.DEFAULT_TOP_K, filters: Optional[Dict[str, List[str]]]=None, exclude: Optional[Iterable[Any]]=None) -> List[Dict[str, Any]]:
        self.load_metadata()
        self.load_index_mapping()
        name_map = {'resnet': 'resnet50', 'effnet': 'efficientnet_b0', 'clip': 'clip_image'}
        emb_name = name_map[model]
        qvec = self._lookup_image_embedding_in_corpus(img_path, emb_name)
        if qvec is None:
            if model not in self._img_embedders_cache:
                self._img_embedders_cache[model] = get_image_embedder(model)
            embedder = self._img_embedders_cache[model]
            qvec = embedder.embed(img_path)
        blocked = {str(item) for item in (exclude or ())}
        limit = max(1, min(int(top_k), RETRIEVAL.DEFAULT_MAX_K))
        # Over-fetch by the number of excluded ids so a blocked self-match does not
        # eat one of the requested slots.
        request_k = limit + len(blocked)
        corpus = self.get_embedding(emb_name)
        dists, idxs = self._search_in_mask(emb_name, corpus, qvec, request_k, self.filter_mask(filters))
        sims = 1.0 - dists
        results = [self._build_result_row(int(i), float(s)) for i, s in zip(idxs, sims)]
        results = [row for row in results if str(row.get('id')) not in blocked]
        return sorted(results, key=lambda r: r['similarity'], reverse=True)[:limit]

    def search_by_text(self, text: str, model: Literal['tfidf', 'sbert', 'clip']=TEXT_MODELS.DEFAULT, top_k: int=RETRIEVAL.DEFAULT_TOP_K, filters: Optional[Dict[str, List[str]]]=None, attribute_boost: bool=True) -> List[Dict[str, Any]]:
        self.load_metadata()
        self.load_index_mapping()
        emb_name = {'tfidf': 'tfidf', 'sbert': 'sbert', 'clip': 'clip_text'}[model]
        if model not in self._txt_embedders_cache:
            if model == 'tfidf' and PATHS.TFIDF_VECTORIZER.exists():
                self._txt_embedders_cache[model] = TFIDFEmbedder.load_fitted()
            else:
                self._txt_embedders_cache[model] = get_text_embedder(model)
        embedder = self._txt_embedders_cache[model]
        corrected_text, fixes = _autocorrect_query(text)
        if fixes and getattr(self, 'verbose', False):
            warnings.warn(f'Query autocorrected: {"; ".join(fixes)}', UserWarning)
        self.last_query_original = text
        self.last_query_corrected = corrected_text
        self.last_query_fixes = fixes
        want_color, want_cat, want_gender, want_style, want_material = parse_query_attrs(corrected_text)
        text_for_expand = corrected_text
        if want_gender and want_gender not in _expand_vn_query(text_for_expand).lower():
            text_expanded = _expand_vn_query(text_for_expand) + f' {want_gender}'
        else:
            text_expanded = _expand_vn_query(text_for_expand)
        qvec = embedder.embed(text_expanded)
        corpus = self.get_embedding(emb_name)
        if model == 'tfidf':
            if qvec.shape[0] < corpus.shape[1]:
                qp = np.zeros(corpus.shape[1], dtype=np.float32)
                qp[:qvec.shape[0]] = qvec
                qvec = qp
            elif qvec.shape[0] > corpus.shape[1]:
                qvec = qvec[:corpus.shape[1]]
            n = np.linalg.norm(qvec) or 1.0
            qvec = qvec / n
        mask = self.filter_mask(filters)
        if mask is not None and not mask.any():
            return []
        if want_color or want_cat:
            qn = qvec / (np.linalg.norm(qvec) or 1.0)
            candidate_idx = np.arange(len(corpus)) if mask is None else np.flatnonzero(mask)
            candidate_corpus = corpus[candidate_idx]
            cn = candidate_corpus / (np.linalg.norm(candidate_corpus, axis=1, keepdims=True) + 1e-09)
            idxs = candidate_idx
            sims = (cn @ qn.astype(np.float32)).astype(float)
        else:
            pool_size = len(corpus) if mask is None else int(mask.sum())
            pool_k = min(pool_size, max(top_k * 10, 200))
            dists, idxs = self._search_in_mask(emb_name, corpus, qvec, pool_k, mask)
            sims = 1.0 - dists
        md = self.load_metadata()
        has_gender_col = 'gender' in md.columns
        scored = []
        for i, s in zip(idxs, sims):
            boost = 0.0
            row = md.iloc[int(i)]
            row_color = str(row.get('color', '')).lower()
            col_exact = bool(want_color) and row_color == want_color
            col_near = bool(want_color) and (not col_exact) and row_color in _COLOR_SIMILAR.get(want_color, ())
            if col_exact:
                boost += 0.6
            elif col_near:
                boost += 0.35
            if want_cat and CATEGORIES.normalize(str(row.get('category', ''))) == want_cat:
                boost += 0.4
            if want_style and str(row.get('style', '')).lower() == want_style:
                boost += 0.25
            if want_material and str(row.get('material', '')).lower() == want_material:
                boost += 0.15
            if want_gender and has_gender_col and (str(row.get('gender', '')).lower() == want_gender):
                boost += 0.2
            scored.append((float(s) + boost if attribute_boost else float(s), int(i), float(s)))
        scored.sort(reverse=True)

        def _key(t):
            row = md.iloc[t[1]]
            row_color = str(row.get('color', '')).lower()
            col_exact = bool(want_color) and row_color == want_color
            col_near = bool(want_color) and (not col_exact) and row_color in _COLOR_SIMILAR.get(want_color, ())
            col_fam = col_exact or col_near
            cat_ok = bool(want_cat) and CATEGORIES.normalize(str(row.get('category', ''))) == want_cat
            sty_ok = bool(want_style) and str(row.get('style', '')).lower() == want_style
            mat_ok = bool(want_material) and str(row.get('material', '')).lower() == want_material
            gen_ok = bool(want_gender) and has_gender_col and (str(row.get('gender', '')).lower() == want_gender)
            # Category dominates: asking for a shirt must not surface a dress just
            # because the dress happens to be the exact colour requested.
            return (cat_ok, col_exact, col_fam, sty_ok, mat_ok, gen_ok, t[0])
        if attribute_boost and (want_cat or want_color or want_style or want_material or (want_gender and has_gender_col)):
            scored.sort(key=_key, reverse=True)
        results = []
        for rank_score, i, s in scored[:top_k]:
            r = self._build_result_row(i, s)
            r['similarity'] = float(max(0.0, min(1.0, s)))
            r['rank_score'] = float(rank_score)
            results.append(r)
        return results

    def search_multimodal(self, img_path: Optional[str | Path]=None, text: Optional[str]=None, alpha: float=RETRIEVAL.FUSION_ALPHA_DEFAULT, top_k: int=RETRIEVAL.DEFAULT_TOP_K, filters: Optional[Dict[str, List[str]]]=None, attribute_boost: bool=False) -> List[Dict[str, Any]]:
        self.load_metadata()
        self.load_index_mapping()
        components: List[Tuple[float, np.ndarray]] = []
        if img_path is not None:
            w_img = max(0.0, min(1.0, 1.0 - alpha))
            if w_img > 0:
                v = self._lookup_image_embedding_in_corpus(img_path, 'clip_image')
                if v is None:
                    if 'clip' not in self._img_embedders_cache:
                        self._img_embedders_cache['clip'] = get_image_embedder('clip')
                    v = self._img_embedders_cache['clip'].embed(img_path)
                components.append((w_img, v))
        if text and len(text.strip()) > 0:
            w_txt = max(0.0, min(1.0, alpha))
            if w_txt > 0:
                if 'clip' not in self._txt_embedders_cache:
                    self._txt_embedders_cache['clip'] = get_text_embedder('clip')
                v = self._txt_embedders_cache['clip'].embed(text)
                components.append((w_txt, v))
        if not components:
            raise ValueError('multimodal needs at least img_path OR text')
        dim = components[0][1].shape[0]
        q = np.zeros(dim, dtype=np.float32)
        total_w = 0.0
        for w, v in components:
            q += w * np.asarray(v, dtype=np.float32)
            total_w += w
        if total_w > 0:
            q = q / total_w
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        corpus = self.get_embedding('clip_image')
        # Over-fetch candidates so attribute boost can surface exact-color items
        # that are deeper in the ANN ranking (same fix as search_by_text).
        fetch_k = top_k
        if attribute_boost and text:
            fetch_k = min(len(corpus), max(top_k * 10, 200))
        dists, idxs = self._search_in_mask('clip_image', corpus, q, fetch_k, self.filter_mask(filters))
        sims = 1.0 - dists

        # Optional text-attribute boost (used by the storefront search; disabled
        # for benchmark model comparison so embeddings stay comparable).
        corrected_text = text or ''
        want_color = want_cat = want_style = want_material = want_gender = None
        if attribute_boost and text:
            corrected_text, _ = _autocorrect_query(text)
            want_color, want_cat, want_gender, want_style, want_material = parse_query_attrs(corrected_text)

        md = self.load_metadata()
        has_gender_col = 'gender' in md.columns
        if attribute_boost and text and (want_color or want_cat or want_style or want_material or (want_gender and has_gender_col)):
            scored = []
            for i, s in zip(idxs, sims):
                boost = 0.0
                row = md.iloc[int(i)]
                row_color = str(row.get('color', '')).lower()
                col_exact = bool(want_color) and row_color == want_color
                col_near = bool(want_color) and (not col_exact) and row_color in _COLOR_SIMILAR.get(want_color, ())
                if col_exact:
                    boost += 0.6
                elif col_near:
                    boost += 0.35
                if want_cat and CATEGORIES.normalize(str(row.get('category', ''))) == want_cat:
                    boost += 0.4
                if want_style and str(row.get('style', '')).lower() == want_style:
                    boost += 0.25
                if want_material and str(row.get('material', '')).lower() == want_material:
                    boost += 0.15
                if want_gender and has_gender_col and (str(row.get('gender', '')).lower() == want_gender):
                    boost += 0.2
                scored.append((float(s) + boost, int(i), float(s)))
            scored.sort(reverse=True)

            def _key(t):
                row = md.iloc[t[1]]
                row_color = str(row.get('color', '')).lower()
                col_exact = bool(want_color) and row_color == want_color
                col_near = bool(want_color) and (not col_exact) and row_color in _COLOR_SIMILAR.get(want_color, ())
                col_fam = col_exact or col_near
                cat_ok = bool(want_cat) and CATEGORIES.normalize(str(row.get('category', ''))) == want_cat
                sty_ok = bool(want_style) and str(row.get('style', '')).lower() == want_style
                mat_ok = bool(want_material) and str(row.get('material', '')).lower() == want_material
                gen_ok = bool(want_gender) and has_gender_col and (str(row.get('gender', '')).lower() == want_gender)
                return (cat_ok, col_exact, col_fam, sty_ok, mat_ok, gen_ok, t[0])

            scored.sort(key=_key, reverse=True)
            results = []
            for rank_score, i, s in scored[:top_k]:
                r = self._build_result_row(i, s)
                r['similarity'] = float(max(0.0, min(1.0, s)))
                r['rank_score'] = float(rank_score)
                results.append(r)
            return results

        results = [self._build_result_row(int(i), float(s)) for i, s in zip(idxs, sims)]
        return sorted(results, key=lambda r: r['similarity'], reverse=True)

    def outfit_match(self, product_id: Any, target_category: str, top_k: int=RETRIEVAL.DEFAULT_TOP_K, match_mode: str='tone_on_tone', filters: Optional[Dict[str, List[str]]]=None) -> List[Dict[str, Any]]:
        self.load_metadata()
        self.load_index_mapping()
        df = self.metadata
        src_mask = df.id.astype(str) == str(product_id)
        if not src_mask.any():
            try:
                src_mask = df.id == int(product_id)
            except:
                pass
        if not src_mask.any():
            raise ValueError(f'product_id {product_id} not found')
        src = df[src_mask].iloc[0]
        src_cat = CATEGORIES.normalize(src.get('category', 'other'))
        src_color = str(src.get('color', '')).strip()
        src_material = str(src.get('material', '')).strip()
        src_style = str(src.get('style', '')).strip()
        tgt_cat_norm = CATEGORIES.normalize(target_category)
        tgt_mask = (df.category.apply(CATEGORIES.normalize) == tgt_cat_norm) & (df.id.astype(str) != str(product_id))
        facet_mask = self.filter_mask(filters)
        if facet_mask is not None:
            tgt_mask = tgt_mask & facet_mask
        if RETRIEVAL.OUTFIT_MATCH_FILTER_CATEGORY:
            sub_df = df[tgt_mask].reset_index()
        else:
            sub_df = df.reset_index()
        if len(sub_df) == 0:
            return []
        sub_indices_global = sub_df['index'].tolist()
        qtext = f'{src_color} {src_material} {src_style} clothing for {src_cat} and {tgt_cat_norm}'
        if 'sbert' not in self._txt_embedders_cache:
            self._txt_embedders_cache['sbert'] = get_text_embedder('sbert')
        emb = self._txt_embedders_cache['sbert']
        qvec = emb.embed(qtext)
        corpus_all = self.get_embedding('sbert')
        sub_corpus = np.asarray([corpus_all[i] for i in sub_indices_global], dtype=np.float32)
        # Over-fetch candidates so the color boost can actually pull same-color
        # items up: previously k == top_k meant boosting only reordered items
        # SBERT already returned, and same-color products deeper in the ranking
        # never surfaced (the "not working" symptom).
        pool_k = min(len(sub_corpus), max(int(top_k) * 10, 50))
        k = max(1, pool_k)
        nn = NearestNeighbors(n_neighbors=k, metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1).fit(sub_corpus)
        dists, sub_idxs = nn.kneighbors(np.atleast_2d(qvec), n_neighbors=k)
        dists = dists[0]
        sub_idxs = sub_idxs[0]
        src_color_low = (src_color or '').lower()
        # Fallback: metadata color missing -> sample dominant color from image.
        if not src_color_low or src_color_low in ('', 'other', 'unknown', 'none'):
            try:
                from src.dataset_adaptor import extract_color_from_image
                guess = extract_color_from_image(str(src.get('image_path', '')))
                if guess:
                    src_color = str(guess)
                    src_color_low = src_color.lower()
            except Exception:
                pass
        src_similar = set(_COLOR_SIMILAR.get(src_color_low, ()))
        _COMPLEMENTARY: Dict[str, tuple] = {
            'green': ('red', 'maroon', 'pink'), 'red': ('green', 'teal', 'mint'),
            'blue': ('orange', 'mustard', 'gold'), 'orange': ('blue', 'navy', 'teal'),
            'yellow': ('purple', 'navy', 'charcoal'), 'purple': ('yellow', 'mustard', 'gold'),
            'black': ('white', 'cream', 'beige'), 'white': ('black', 'navy', 'charcoal'),
            'navy': ('beige', 'cream', 'mustard'), 'beige': ('navy', 'brown', 'olive'),
            'pink': ('olive', 'green', 'navy'), 'brown': ('cream', 'beige', 'cyan'),
            'olive': ('maroon', 'burgundy', 'navy'), 'teal': ('coral', 'red', 'mustard'),
        }
        want_complement = set(_COMPLEMENTARY.get(src_color_low, ()))
        results: List[Dict[str, Any]] = []
        for d, si in zip(dists, sub_idxs):
            global_idx = sub_indices_global[si]
            sim = 1.0 - float(d)
            row = self._build_result_row(int(global_idx), sim)
            tgt_color = str(row.get('color', '') or '')
            tgt_low = tgt_color.lower()
            # Color-match boost: exact same color first, then same color family.
            # e.g. green shirt -> green shoes/pants rank top, olive/mint/teal next.
            color_boost = 0.0
            color_note = ''
            if src_color_low and tgt_low:
                if match_mode == 'contrast':
                    if tgt_low in want_complement:
                        color_boost = 0.5
                        color_note = f'Màu {tgt_color} tương phản {src_color} — phối nổi bật'
                    elif tgt_low == src_color_low:
                        color_boost = 0.15
                        color_note = f'Cùng màu {src_color} — phối ton-sur-ton'
                    elif tgt_low in src_similar:
                        color_boost = 0.05
                elif tgt_low == src_color_low:
                    color_boost = 0.5
                    color_note = f'Cùng màu {src_color} — phối ton-sur-ton'
                elif tgt_low in src_similar:
                    color_boost = 0.25
                    color_note = f'Màu {tgt_color} cùng tông {src_color}'
            sim = float(max(0.0, min(1.0, sim + color_boost)))
            row['similarity'] = sim
            # Tier for tiered ranking: 0 = exact color, 1 = same family /
            # complementary, 2 = no color relation. Tier dominates SBERT score
            # so same-color items always surface first.
            if match_mode == 'contrast':
                tier = 0 if (tgt_low and tgt_low in want_complement) else (1 if (tgt_low == src_color_low or tgt_low in src_similar) else 2)
            else:
                tier = 0 if (tgt_low and src_color_low and tgt_low == src_color_low) else (1 if tgt_low in src_similar else 2)
            row['_color_tier'] = tier
            reasons = []
            tgt_material = row.get('material', '')
            tgt_style = row.get('style', '')
            if color_note:
                reasons.append(color_note)
            if src_material and tgt_material and (src_material.lower() == tgt_material.lower()):
                reasons.append(f'Cùng chất liệu {src_material}')
            if src_style and tgt_style and (src_style.lower() == tgt_style.lower()):
                reasons.append(f'Phong cách {src_style} tương đồng')
            if not reasons:
                reasons.append('Tương đồng về phong cách thời trang tổng thể')
            row['explanation'] = reasons
            results.append(row)
        # Tiered sort: color tier first (exact > family > rest), then SBERT+boost
        # score inside each tier. Guarantees same-color items top the list.
        results = sorted(results, key=lambda r: (r.pop('_color_tier', 2), -r['similarity']))
        return results[:max(1, int(top_k))]

