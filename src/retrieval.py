import os, pickle, warnings
from pathlib import Path, PureWindowsPath
from typing import Literal, Any, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from config import PATHS, IMAGE_MODELS, TEXT_MODELS, RETRIEVAL, CATEGORIES, DEVICE, CLIP as CLIP_CFG
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
    """Tim anh danh muc trong thu muc du an hien tai khi chuyen may."""
    stored_path = str(image_path).strip()
    # PureWindowsPath doc duoc ca duong dan Windows va ten dung dau '/'.
    # Dung ten file de tim lai anh trong ban du an vua giai nen.
    filename = PureWindowsPath(stored_path).name
    local_path = PATHS.DATA_PROCESSED_IMAGES / filename
    if filename and local_path.is_file():
        return local_path.resolve()

    # Van ho tro anh o vi tri khac neu duong dan da luu con hop le.
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
            self.image_paths = [_resolve_catalog_image_path(p) for p in m['image_paths']]
        return (self.product_ids, self.image_paths)

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

    def _kneighbors(self, emb_name: str, corpus: np.ndarray, query_vec: np.ndarray, top_k: int, extra_neighbors: int=0) -> Tuple[np.ndarray, np.ndarray]:
        if len(corpus) == 0:
            return (np.array([], dtype=np.float32), np.array([], dtype=np.int64))
        q = np.atleast_2d(np.asarray(query_vec, dtype=np.float32))
        topk = min(len(corpus), max(1, min(int(top_k), RETRIEVAL.DEFAULT_MAX_K)) + max(0, int(extra_neighbors)))
        nn = self._get_or_fit_nn(emb_name, corpus)
        if topk > nn.n_neighbors:
            nn2 = NearestNeighbors(n_neighbors=topk, metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1).fit(corpus)
            self._nn_index_cache[emb_name] = nn2
            nn = nn2
        dists, idxs = nn.kneighbors(q, n_neighbors=topk)
        return (dists[0], idxs[0])

    def _build_result_row(self, idx: int, sim: float) -> Dict[str, Any]:
        md = self.load_metadata().iloc[idx]
        return {'index': int(idx), 'id': self.product_ids[idx] if self.product_ids else int(md.get('id', idx)), 'image_path': str(self.image_paths[idx]) if self.image_paths else str(md.image_path), 'product_name': str(md.get('product_name', '')), 'description': str(md.get('description', '')), 'category': str(md.get('category', 'other')), 'color': str(md.get('color', '')), 'material': str(md.get('material', '')), 'style': str(md.get('style', '')), 'similarity': float(max(0.0, min(1.0, sim)))}

    def _lookup_image_index_in_corpus(self, img_path) -> Optional[int]:
        try:
            p = Path(img_path).resolve()
        except Exception:
            p = str(img_path)
        if self.image_paths is None:
            return None
        for i, ip in enumerate(self.image_paths):
            try:
                if Path(ip).resolve() == p:
                    return i
            except Exception:
                try:
                    if str(ip) == str(img_path):
                        return i
                except Exception:
                    pass
        return None

    def _lookup_image_embedding_in_corpus(self, img_path, emb_name: str) -> Optional[np.ndarray]:
        idx = self._lookup_image_index_in_corpus(img_path)
        if idx is None:
            return None
        return np.asarray(self.get_embedding(emb_name)[idx], dtype=np.float32).copy()

    def search_by_image(self, img_path: str | Path, model: Literal['resnet', 'effnet', 'clip']=RETRIEVAL.DEFAULT_IMAGE, top_k: int=RETRIEVAL.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        self.load_metadata()
        self.load_index_mapping()
        name_map = {'resnet': 'resnet50', 'effnet': 'efficientnet_b0', 'clip': 'clip_image'}
        emb_name = name_map[model]
        corpus = self.get_embedding(emb_name)
        query_idx = self._lookup_image_index_in_corpus(img_path)
        if len(corpus) <= (1 if query_idx is not None else 0):
            return []
        if query_idx is not None:
            qvec = np.asarray(corpus[query_idx], dtype=np.float32).copy()
        else:
            if model not in self._img_embedders_cache:
                self._img_embedders_cache[model] = get_image_embedder(model)
            embedder = self._img_embedders_cache[model]
            qvec = embedder.embed(img_path)
        limit = max(1, min(int(top_k), RETRIEVAL.DEFAULT_MAX_K))
        # Lay them mot ung vien de van du top-k sau khi loai anh truy van.
        dists, idxs = self._kneighbors(emb_name, corpus, qvec, limit, extra_neighbors=1 if query_idx is not None else 0)
        sims = 1.0 - dists
        results = [self._build_result_row(int(i), float(s)) for i, s in zip(idxs, sims) if int(i) != query_idx]
        return sorted(results, key=lambda r: r['similarity'], reverse=True)[:limit]

    def search_by_text(self, text: str, model: Literal['tfidf', 'sbert', 'clip']=TEXT_MODELS.DEFAULT, top_k: int=RETRIEVAL.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
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
            if qvec.shape[0] != corpus.shape[1]:
                raise ValueError('TF-IDF vectorizer và vector danh mục không khớp. Dừng ứng dụng rồi chạy: py -3.11 scripts/rebuild_tfidf.py')
            n = float(np.linalg.norm(qvec))
            if not np.isfinite(n) or n == 0.0:
                raise ValueError('Không có từ phù hợp trong bộ từ vựng TF-IDF. Hãy thử mô tả khác; nếu bộ từ vựng chưa được xây từ danh mục hiện tại, dừng ứng dụng rồi chạy: py -3.11 scripts/rebuild_tfidf.py')
            qvec = qvec / n
        if want_color or want_cat:
            qn = qvec / (np.linalg.norm(qvec) or 1.0)
            cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-09)
            sims_all = (cn @ qn.astype(np.float32)).astype(float)
            idxs = np.arange(len(corpus))
            sims = sims_all
        else:
            pool_k = min(len(corpus), max(top_k * 10, 200))
            dists, idxs = self._kneighbors(emb_name, corpus, qvec, pool_k)
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
            return (col_exact, col_fam, cat_ok, sty_ok, mat_ok, gen_ok, t[0])
        if want_cat or want_color or want_style or want_material or (want_gender and has_gender_col):
            scored.sort(key=_key, reverse=True)
        results = []
        for b, i, s in scored[:top_k]:
            r = self._build_result_row(i, s)
            r['similarity'] = float(max(0.0, min(1.0, s)))
            results.append(r)
        return results

    def search_multimodal(self, img_path: Optional[str | Path]=None, text: Optional[str]=None, alpha: float=RETRIEVAL.FUSION_ALPHA_DEFAULT, top_k: int=RETRIEVAL.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
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
        dists, idxs = self._kneighbors('clip_image', corpus, q, top_k)
        sims = 1.0 - dists
        results = [self._build_result_row(int(i), float(s)) for i, s in zip(idxs, sims)]
        return sorted(results, key=lambda r: r['similarity'], reverse=True)

    def outfit_match(self, product_id: Any, target_category: str, top_k: int=RETRIEVAL.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
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
        k = min(max(1, top_k), len(sub_corpus))
        nn = NearestNeighbors(n_neighbors=k, metric=RETRIEVAL.METRIC, algorithm=RETRIEVAL.ALGORITHM, n_jobs=1).fit(sub_corpus)
        dists, sub_idxs = nn.kneighbors(np.atleast_2d(qvec), n_neighbors=k)
        dists = dists[0]
        sub_idxs = sub_idxs[0]
        results: List[Dict[str, Any]] = []
        for d, si in zip(dists, sub_idxs):
            global_idx = sub_indices_global[si]
            sim = 1.0 - float(d)
            row = self._build_result_row(int(global_idx), sim)
            reasons = []
            tgt_color = row.get('color', '')
            tgt_material = row.get('material', '')
            tgt_style = row.get('style', '')
            if src_color and tgt_color and (src_color.lower() == tgt_color.lower()):
                reasons.append(f'Cùng màu {src_color}')
            if src_material and tgt_material and (src_material.lower() == tgt_material.lower()):
                reasons.append(f'Cùng chất liệu {src_material}')
            if src_style and tgt_style and (src_style.lower() == tgt_style.lower()):
                reasons.append(f'Phong cách {src_style} tương đồng')
            if not reasons:
                reasons.append('Tương đồng về phong cách thời trang tổng thể')
            row['explanation'] = reasons
            results.append(row)
        return sorted(results, key=lambda r: r['similarity'], reverse=True)


def tune_multimodal_alpha(retriever, query_items, alphas=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0), top_k=10):
    from src import metrics as M
    best = (0.5, -1.0)
    rows = []
    for a in alphas:
        aps = []
        for q in query_items:
            res = retriever.search_multimodal(img_path=q.get('img'), text=q.get('text'), alpha=a, top_k=top_k)
            pred = [x['id'] for x in res]
            aps.append(M.average_precision_at_k(pred, q.get('relevant', []), top_k))
        m = float(sum(aps) / max(1, len(aps)))
        rows.append({'alpha': a, 'map': m})
        if m > best[1]:
            best = (a, m)
    return {'best_alpha': best[0], 'best_map': best[1], 'rows': rows}
