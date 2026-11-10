from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
from PIL import Image

from config import PATHS
from src.filters import FACET_LABELS, facet_options, filter_summary
from src.user_store import UserStore

FACET_ORDER: tuple = ('category', 'color', 'material', 'style', 'gender')
def load_image_safe(img_path: str | Path, size: tuple | None = None, thumb: tuple = (600, 600)) -> Image.Image | None:
    try:
        img = Image.open(img_path).convert('RGB')
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
            return img
        img.thumbnail(thumb, Image.LANCZOS)
        return img
    except Exception:
        return None


STORE_CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1320px;}
  .ff-brand {font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: 0;}
  .ff-tagline {color: #6b7280; font-size: .82rem; margin-top: -.2rem;}
  .ff-chip {display:inline-block; padding: 1px 8px; margin: 0 4px 4px 0; border-radius: 999px;
            background: #f1f3f6; color: #374151; font-size: .72rem; text-transform: capitalize;}
  .ff-price {font-weight: 600; font-size: .95rem; color: #111827;}
  .ff-muted {color: #6b7280; font-size: .78rem;}
  .ff-hero {padding: 1.1rem 1.3rem; border-radius: 14px; background: linear-gradient(120deg,#111827,#374151);
            color: #f9fafb; margin-bottom: .6rem;}
  .ff-hero h2 {margin: 0 0 .2rem 0; font-size: 1.25rem;}
  .ff-hero p {margin: 0; color: #d1d5db; font-size: .85rem;}
  .ff-shelf-title {font-size: 1rem; font-weight: 600; margin: .3rem 0 .5rem 0;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
  .stButton button {border-radius: 8px;}
</style>
"""


def inject_css() -> None:
    st.markdown(STORE_CSS, unsafe_allow_html=True)


def chips_html(product: Dict[str, Any]) -> str:
    parts = [
        str(product.get(field, '')).strip()
        for field in ('category', 'color', 'material', 'style')
        if str(product.get(field, '')).strip()
    ]
    return ''.join(f'<span class="ff-chip">{part}</span>' for part in parts)


def brand_header(title: str = 'FashionFinder', tagline: str = 'Tìm và phối đồ thời trang bằng hình ảnh, mô tả hoặc cả hai.') -> None:
    st.markdown(f'<p class="ff-brand">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="ff-tagline">{tagline}</p>', unsafe_allow_html=True)

def favorite_toggle(store: UserStore, product_id: Any, key: str, width: str = 'stretch') -> None:
    active = store.is_favorite(product_id)
    label = '❤️ Đã lưu' if active else '♡ Lưu'
    if st.button(label, key=key, width=width):
        store.toggle_favorite(product_id)
        st.rerun()


def product_card(product: Dict[str, Any], store: UserStore, key_prefix: str, show_match: bool = True, on_detail_key: str = 'open_detail') -> None:
    with st.container(border=True):
        image_path = str(product.get('image_path', ''))
        if image_path and Path(image_path).exists():
            st.image(image_path, width=200)
        else:
            placeholder = load_image_safe(image_path)
            if placeholder is not None:
                st.image(placeholder, width='stretch')
            else:
                st.markdown('<div class="ff-muted">Không có ảnh</div>', unsafe_allow_html=True)
        st.markdown(f'**{str(product.get("product_name", "")).strip() or "Sản phẩm"}**')
        st.markdown(chips_html(product), unsafe_allow_html=True)
        if show_match and 'similarity' in product:
            score = float(product.get('similarity', 0.0))
            st.progress(min(1.0, max(0.0, score)), text=f'Độ khớp {score:.0%}')
        cols = st.columns([1, 1])
        with cols[0]:
            favorite_toggle(store, product.get('id', ''), key=f'{key_prefix}_fav_{product.get("id", "")}', width='stretch')
        with cols[1]:
            if st.button('Chi tiết', key=f'{key_prefix}_open_{product.get("id", "")}', width='stretch'):
                st.session_state[on_detail_key] = str(product.get('id', ''))
                st.rerun()


def render_results(results: Sequence[Dict[str, Any]], store: UserStore, cols: int = 4, show_match: bool = True, key_prefix: str = 'res', empty_text: str = 'Không tìm thấy sản phẩm nào phù hợp. Thử bỏ bớt bộ lọc hoặc mô tả khác.') -> None:
    if not results:
        st.info(empty_text)
        return
    for start in range(0, len(results), cols):
        row = st.columns(cols)
        for offset, product in enumerate(results[start:start + cols]):
            with row[offset]:
                product_card(product, store, key_prefix=f'{key_prefix}_{start + offset}', show_match=show_match)


def render_compact_row(results: Sequence[Dict[str, Any]], store: UserStore, key_prefix: str, cols: int = 4, allow_detail: bool = True) -> None:
    if not results:
        st.caption('Chưa có gợi ý.')
        return
    row = st.columns(cols)
    for offset, product in enumerate(results[:cols]):
        with row[offset]:
            image_path = str(product.get('image_path', ''))
            if image_path and Path(image_path).exists():
                st.image(image_path, width=200)
            st.caption(f"{str(product.get('product_name', ''))[:38]}")
            if allow_detail:
                actions = st.columns([1, 1])
            else:
                actions = [st.container()]
            with actions[0]:
                favorite_toggle(store, product.get('id', ''), key=f'{key_prefix}_fav_{product.get("id", "")}')
            if allow_detail:
                with actions[1]:
                    if st.button('Chi tiết', key=f'{key_prefix}_open_{product.get("id", "")}', width='stretch'):
                        st.session_state['detail_pid'] = str(product.get('id', ''))
                        st.rerun()


def render_product_grid(results: List[Dict[str, Any]], cols: int = 5, caption_template: str = '{product_name}\n{category} | {color} | sim={similarity:.3f}', show_explanation: bool = False) -> None:
    if not results:
        st.info('Không có kết quả nào.')
        return
    for i in range(0, len(results), cols):
        batch = results[i:i + cols]
        row_cols = st.columns(cols)
        for j, result in enumerate(batch):
            with row_cols[j]:
                img_path = str(result.get('image_path', ''))
                if img_path and Path(img_path).exists():
                    try:
                        st.image(img_path, width='stretch')
                    except Exception:
                        st.write('⚠️ no image')
                else:
                    img = load_image_safe(img_path)
                    if img is not None:
                        st.image(img, width='stretch')
                    else:
                        st.write('⚠️ no image')
                safe = {
                    'product_name': str(result.get('product_name', '')),
                    'category': str(result.get('category', '')),
                    'color': str(result.get('color', '')),
                    'material': str(result.get('material', '')),
                    'style': str(result.get('style', '')),
                    'similarity': float(result.get('similarity', 0.0)),
                    'id': str(result.get('id', '')),
                    'description': str(result.get('description', '')),
                }
                try:
                    st.caption(caption_template.format(**safe))
                except Exception:
                    st.caption(f"{safe['product_name']} | sim={safe['similarity']:.3f}")
                if show_explanation and result.get('explanation'):
                    st.caption('💡 ' + '; '.join(result['explanation']))

def render_facet_panel(catalog: Any, filters: Optional[Dict[str, Any]], key_prefix: str = 'facet', total: Optional[int] = None) -> Dict[str, List[str]]:
    filters = dict(filters or {})
    counts = catalog.facets(filters)
    updated: Dict[str, List[str]] = {}
    shown = 0
    for column in FACET_ORDER:
        options = facet_options(counts, column)
        if not options:
            continue
        shown += 1
        selected = [value for value in (filters.get(column) or []) if value in options]
        labels = {value: f'{value} ({count})' for value, count in counts.get(column, [])}
        matching = sum(count for _, count in counts.get(column, []))
        label = FACET_LABELS.get(column, column)
        if total is not None and matching < int(total):
            label = f'{label} · {matching}'
        updated[column] = st.multiselect(
            label,
            options=options,
            default=selected,
            format_func=lambda value, labels=labels: labels.get(value, value),
            key=f'{key_prefix}_{column}',
            placeholder='Tất cả',
        )
    if shown == 0:
        st.caption('Bộ lọc chưa sẵn sàng — hãy rebuild dữ liệu.')
    return updated


def render_active_filters(filters: Optional[Dict[str, Any]], key_prefix: str = 'active') -> None:
    summary = filter_summary(filters)
    if not summary:
        return
    columns = st.columns([3, 1])
    with columns[0]:
        st.markdown(' '.join(f'<span class="ff-chip">{label}</span>' for label in summary), unsafe_allow_html=True)
    with columns[1]:
        if st.button('Xóa bộ lọc', key=f'{key_prefix}_clear', width='stretch'):
            for column in FACET_ORDER:
                st.session_state.pop(f'{key_prefix.replace("active", "facet")}_{column}', None)
            st.session_state['filters'] = {}
            st.rerun()


def _use_query(state_key: str, query: str) -> None:
    st.session_state[state_key] = query
    st.session_state['run_search'] = True


def render_recent_searches(store: UserStore, on_select_key: str = 'query_input', key_prefix: str = 'recent', limit: int = 5) -> None:
    history = store.recent_searches[:limit]
    if not history:
        return
    st.caption('Tìm kiếm gần đây')
    columns = st.columns(len(history))
    for index, entry in enumerate(history):
        query = str(entry.get('query', ''))
        with columns[index]:
            st.button(query[:28], key=f'{key_prefix}_{index}', width='stretch', on_click=_use_query, args=(on_select_key, query))
    if st.button('Xóa lịch sử tìm kiếm', key=f'{key_prefix}_clear'):
        store.clear_searches()
        st.rerun()


def status_strip(catalog: Any, device: str, extra: str = '') -> None:
    df = catalog.df
    parts = [f'**{len(df)}** sản phẩm']
    for column in ('category', 'color', 'material'):
        if column in df.columns:
            parts.append(f'{df[column].nunique()} {FACET_LABELS.get(column, column).lower()}')
    if extra:
        parts.append(extra)
    st.caption(f"Device `{device}` · " + ' · '.join(parts))


def empty_state(message: str, hint: str = '') -> None:
    st.markdown(f'<div class="ff-hero"><h2>{message}</h2><p>{hint}</p></div>', unsafe_allow_html=True)

@st.dialog('Chi tiết sản phẩm', width='large')
def show_product_detail(catalog: Any, store: UserStore, product_id: str, similarity: Optional[float] = None) -> None:
    product = catalog.get(product_id)
    if product is None:
        st.error('Không tìm thấy sản phẩm này.')
        return
    store.record_view(product_id)
    left, right = st.columns([1, 1])
    with left:
        image_path = str(product.get('image_path', ''))
        if image_path and Path(image_path).exists():
            st.image(image_path, width=200)
    with right:
        st.subheader(str(product.get('product_name', 'Sản phẩm')))
        st.markdown(chips_html(product), unsafe_allow_html=True)
        st.caption(f"Mã sản phẩm · {product.get('id', '')}")
        if similarity is not None:
            st.progress(min(1.0, max(0.0, float(similarity))), text=f'Độ khớp với truy vấn {float(similarity):.0%}')
        if product.get('description'):
            with st.expander('Mô tả', expanded=True):
                st.write(str(product['description']))
        favorite_toggle(store, product_id, key=f'detail_fav_{product_id}')
        st.divider()
        targets = catalog.outfit_targets(product_id)
        if targets:
            st.markdown('**Phối cùng sản phẩm này**')
            picked = st.pills('Danh mục', options=targets, default=targets[0], key=f'detail_target_{product_id}', label_visibility='collapsed')
            mode = st.radio('Kiểu phối màu', options=['tone_on_tone', 'contrast'], format_func=lambda m: 'Ton-sur-ton' if m == 'tone_on_tone' else 'Tương phản', horizontal=True, key=f'detail_mode_{product_id}')
            render_compact_row(catalog.complete_the_look(product_id, target_category=picked, mode=mode, top_k=4), store, key_prefix=f'detail_look_{product_id}', allow_detail=False)
    st.divider()
    st.markdown('**Sản phẩm tương tự**')
    render_compact_row(catalog.similar(product_id, top_k=4), store, key_prefix=f'detail_sim_{product_id}', allow_detail=False)


def render_shelf(title: str, products: Sequence[Dict[str, Any]], store: UserStore, key_prefix: str, cols: int = 6) -> None:
    if not products:
        return
    st.markdown(f'<div class="ff-shelf-title">{title}</div>', unsafe_allow_html=True)
    for start in range(0, len(products), cols):
        render_compact_row(products[start:start + cols], store, key_prefix=f'{key_prefix}_{start}', cols=cols)


def stale_banner() -> None:
    from src.dataset_adaptor import is_embedding_stale
    if not PATHS.EMB_MANIFEST.exists():
        st.error('❌ Chưa có embedding nào được trích xuất. Chạy Rebuild Toàn Bộ.')
        return
    if is_embedding_stale(PATHS.EMB_MANIFEST, PATHS.DATA_PROCESSED_METADATA, PATHS.DATA_PROCESSED_IMAGES):
        st.warning('⚠️ Embedding đã cũ (dataset thay đổi). Nhấn **Rebuild dữ liệu** ở khu vực quản trị để cập nhật.')


def _run_pipeline(script: str, extra_args: Sequence[str], spinner_text: str) -> None:
    import subprocess
    import sys

    cmd = [sys.executable, script, *extra_args]
    try:
        with st.spinner(spinner_text):
            result = subprocess.run(cmd, cwd=str(PATHS.PROJECT_ROOT), capture_output=True, text=True)
    except Exception as exc:
        st.error(f'Không chạy được {script}: {exc}')
        return
    if result.returncode == 0:
        st.success(f'`{script}` hoàn tất. Tải lại trang để dùng dữ liệu mới.')
        return
    st.error(f'`{script}` thất bại.')
    if result.stderr:
        with st.expander('Chi tiết lỗi'):
            st.code(result.stderr[-2000:])


def admin_sidebar_components() -> None:
    st.markdown('**Dữ liệu & mô hình**')
    cols = st.columns(2)
    with cols[0]:
        btn_rebuild = st.button('🔄 Rebuild', key='admin_rebuild', width='stretch')
    with cols[1]:
        force_flag = st.checkbox('Force', value=False, key='admin_force')
    if btn_rebuild:
        _run_pipeline('scripts/rebuild_all.py', ['--force'] if force_flag else [], 'Đang rebuild toàn bộ pipeline...')
    if st.button('⚡ Rebuild TF-IDF', key='admin_rebuild_tfidf', width='stretch'):
        _run_pipeline('scripts/rebuild_tfidf.py', [], 'Đang dựng lại TF-IDF vectorizer...')
    st.caption('Rebuild TF-IDF chỉ dựng lại vectorizer + vectors, giữ nguyên các embedding khác.')
