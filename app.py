import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap_env
import streamlit as st

st.set_page_config(page_title='FashionFinder', page_icon='👗', layout='wide', initial_sidebar_state='expanded')

import pandas as pd
from PIL import Image

from config import DEVICE, PATHS, RETRIEVAL
from src.catalog import Catalog
from src.retrieval import Retriever
from src.ui_utils import (
    admin_sidebar_components,
    brand_header,
    empty_state,
    inject_css,
    render_active_filters,
    render_compact_row,
    render_facet_panel,
    render_product_grid,
    render_recent_searches,
    render_results,
    render_shelf,
    show_product_detail,
    stale_banner,
    status_strip,
)
from src.user_store import UserStore, new_session_id

PAGE_SIZE = 12
SORT_OPTIONS = ('Liên quan nhất', 'Tên A→Z', 'Danh mục', 'Màu sắc')
MODEL_OPTIONS = ('Chỉ văn bản', 'Chỉ hình ảnh', 'Hybrid (ảnh + văn bản)')
_MODE_PARAM_TO_LABEL = {'text': MODEL_OPTIONS[0], 'image': MODEL_OPTIONS[1], 'hybrid': MODEL_OPTIONS[2]}
CATEGORIES_NAMES = {
    'shirt': 'Áo',
    'trouser': 'Quần',
    'shoes': 'Giày',
    'bag': 'Túi xách',
    'dress': 'Váy & đầm',
    'watch': 'Đồng hồ',
    'underwear': 'Đồ lót',
    'other': 'Khác',
}
SUGGESTION_QUERIES = (
    'Áo thun cotton đen thể thao',
    'Quần jean xanh cổ điển nam',
    'Giày da nâu trang trọng văn phòng',
    'Túi xách da đen thanh lịch nữ',
    'Váy đầm hồng polyester dự tiệc',
)
SAMPLE_IMAGE_LIMIT = 24
BENCHMARK_COLUMNS = (
    ('CLIP Image', 'image', 'clip'),
    ('ResNet50', 'image', 'resnet'),
    ('EfficientNet-B0', 'image', 'effnet'),
    ('SBERT', 'text', 'sbert'),
    ('CLIP Hybrid', 'hybrid', 'clip'),
)


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever | None:
    try:
        return Retriever(lazy=False, verbose=False)
    except Exception as exc:
        st.session_state['boot_error'] = str(exc)
        return None


@st.cache_resource(show_spinner=False)
def get_catalog(_retriever: Retriever) -> Catalog:
    return Catalog(_retriever)


def get_store(session_id: str) -> UserStore:
    store = st.session_state.get('store')
    if store is None or store.session_id != session_id:
        PATHS.ensure_user_state()
        store = UserStore.load(PATHS.USER_STATE_DIR, session_id)
        st.session_state['store'] = store
    return store


def session_id() -> str:
    incoming = str(st.query_params.get('sid', '')).strip()
    sid = incoming or str(st.session_state.get('sid', '')).strip() or new_session_id()
    st.session_state['sid'] = sid
    if incoming != sid:
        st.query_params['sid'] = sid
    return sid


def _upload_to_temp(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def sample_image_options(catalog: Catalog | None) -> list[str]:
    if catalog is None:
        return []
    paths = [str(Path(path)) for path in catalog.df['image_path']]
    return [path for path in paths if Path(path).exists()]


def _use_sample_image() -> None:
    path = str(st.session_state.get('sample_image', '') or '').strip()
    if not path or not Path(path).exists():
        return
    st.session_state['last_image'] = path
    st.session_state['run_search'] = True


def _use_suggestion(query: str) -> None:
    st.session_state['query_input'] = query
    st.session_state['run_search'] = True


def render_suggestion_chips() -> None:
    st.caption('Gợi ý tìm kiếm')
    columns = st.columns(len(SUGGESTION_QUERIES))
    for index, chip in enumerate(SUGGESTION_QUERIES):
        with columns[index]:
            st.button(chip, key=f'suggest_{index}', width='stretch', on_click=_use_suggestion, args=(chip,))


def render_query_notes(retriever: Retriever | None, query: str) -> None:
    if retriever is None or not (query or '').strip():
        return
    fixes = getattr(retriever, 'last_query_fixes', []) or []
    corrected = str(getattr(retriever, 'last_query_corrected', '') or '')
    if fixes and corrected.strip() and corrected.strip().lower() != query.strip().lower():
        st.info(f'🔍 Ý bạn là **"{corrected}"**? (tự sửa: {", ".join(fixes)})')
    try:
        from src.retrieval import _COLOR_SIMILAR, parse_query_attrs

        color, *_ = parse_query_attrs(corrected or query)
        family = [name for name in _COLOR_SIMILAR.get(color, ()) if name] if color else []
        if family:
            st.caption(f'🎨 Màu gần giống **{color}**: {", ".join(family[:6])} cũng được tính (điểm thấp hơn màu chính xác).')
    except Exception:
        pass


inject_css()
retriever = get_retriever()
sid = session_id()
store = get_store(sid)

catalog: Catalog | None = get_catalog(retriever) if retriever is not None else None
brand_header()

if catalog is None:
    empty_state(
        'Chưa có dữ liệu để bán hàng',
        'Không đọc được metadata/embedding. Bật **Chế độ quản trị** ở thanh bên và chạy Rebuild dữ liệu.',
    )
    with st.sidebar:
        admin_sidebar_components()
    st.stop()

if 'filters' not in st.session_state:
    st.session_state['filters'] = {}
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'visible' not in st.session_state:
    st.session_state['visible'] = PAGE_SIZE
if 'query_input' not in st.session_state:
    st.session_state['query_input'] = str(st.query_params.get('q', ''))


def mode_key() -> str:
    return {
        'Hybrid (ảnh + văn bản)': 'hybrid',
        'Chỉ văn bản': 'text',
        'Chỉ hình ảnh': 'image',
    }[st.session_state.get('mode_label', MODEL_OPTIONS[0])]


def run_search(query: str = '', image_path: str | None = None) -> None:
    query = (query or '').strip()
    filters = dict(st.session_state.get('filters') or {})
    mode = mode_key()
    if not query and not image_path:
        st.session_state['results'] = []
        return
    try:
        with st.spinner('Đang tìm sản phẩm phù hợp…'):
            results = catalog.search(
                query=query or None,
                image_path=image_path,
                mode=mode,
                model_image=st.session_state.get('image_model', 'clip'),
                model_text=st.session_state.get('text_model', RETRIEVAL.DEFAULT_TEXT),
                alpha=float(st.session_state.get('alpha', RETRIEVAL.FUSION_ALPHA_DEFAULT)),
                filters=filters,
                top_k=int(st.session_state.get('top_k', RETRIEVAL.DEFAULT_TOP_K)),
            )
    except Exception as exc:
        st.error(f'Không tìm được kết quả: {exc}')
        return
    st.session_state['results'] = results
    st.session_state['visible'] = PAGE_SIZE
    st.session_state['applied_filters'] = filters
    st.session_state['last_query'] = query
    st.session_state['last_image'] = str(image_path) if image_path else None
    if query:
        store.record_search(query, mode)
    st.query_params['q'] = query
    st.query_params['mode'] = mode


def refresh_on_filter_change() -> None:
    applied = st.session_state.get('applied_filters')
    if applied is None or dict(applied) == dict(st.session_state.get('filters') or {}):
        return
    if not (st.session_state.get('last_query') or st.session_state.get('last_image')):
        return
    run_search(st.session_state.get('last_query') or '', st.session_state.get('last_image'))


with st.sidebar:
    st.markdown('### Bộ lọc')
    filters_now = render_facet_panel(catalog, st.session_state.get('filters'), total=len(catalog.df))
    st.session_state['filters'] = filters_now
    st.divider()
    st.markdown('### Kết quả')
    st.selectbox('Sắp xếp theo', SORT_OPTIONS, key='sort_by')
    st.slider('Số kết quả mỗi lần', min_value=4, max_value=RETRIEVAL.DEFAULT_MAX_K, value=RETRIEVAL.DEFAULT_TOP_K, key='top_k')
    sample_options = sample_image_options(catalog)
    if sample_options:
        with st.expander('Ảnh mẫu'):
            st.caption(f'{len(sample_options)} ảnh trong danh mục')
            scroll = st.container(height=480, border=True)
            with scroll:
                thumb_cols = st.columns(4)
                for index, path in enumerate(sample_options):
                    with thumb_cols[index % 4]:
                        try:
                            st.image(path, width='stretch')
                        except Exception:
                            pass
                        if st.button('Dùng ảnh này', key=f'sample_pick_{index}', width='stretch'):
                            st.session_state['sample_image'] = path
                            _use_sample_image()
    with st.expander('Tuỳ chọn mô hình'):
        st.selectbox('Mô hình ảnh', ['clip', 'resnet', 'effnet'], key='image_model')
        st.selectbox('Mô hình văn bản', ['sbert', 'tfidf', 'clip'], key='text_model')
        st.slider('Trọng số văn bản (alpha)', 0.0, 1.0, float(RETRIEVAL.FUSION_ALPHA_DEFAULT), 0.05, key='alpha')
    st.divider()
    admin_mode = st.toggle('Chế độ quản trị', key='admin_mode')
    if admin_mode:
        admin_sidebar_components()
    st.divider()
    stale_banner()
    status_strip(catalog, DEVICE)

# Benchmark (so sánh text-only / image-only / combined) is a graded deliverable of
# the assignment, so it is always reachable; the admin toggle now only gates the
# rebuild/ops controls, not the model comparison.
navigation = ['Khám phá', 'Tủ đồ', 'Phối đồ', 'Benchmark']
active_page = st.segmented_control('Điều hướng', navigation, default=navigation[0], key='active_page', label_visibility='collapsed') or navigation[0]

with st.container(border=True):
    st.radio('Chế độ tìm', MODEL_OPTIONS, key='mode_label', index=MODEL_OPTIONS.index(_MODE_PARAM_TO_LABEL.get(str(st.query_params.get('mode', 'text')), MODEL_OPTIONS[0])), horizontal=True, label_visibility='collapsed')
    mode_now = st.session_state.get('mode_label', MODEL_OPTIONS[0])
    if True:  # (was st.form; removed batching so uploads preview instantly)
        if mode_now == MODEL_OPTIONS[0]:  # text only
            col_q, col_go = st.columns([7, 1])
            with col_q:
                st.text_input('Tìm kiếm', key='query_input', placeholder='Ví dụ: áo sơ mi trắng công sở cotton, quần jean nam, giày da nâu…', label_visibility='collapsed')
            with col_go:
                submitted = st.button('Tìm', type='primary', key='search_btn', width='stretch')
            uploaded = None
        elif mode_now == MODEL_OPTIONS[1]:  # image only
            col_img, col_go = st.columns([7, 1])
            with col_img:
                uploaded = st.file_uploader('Ảnh tham chiếu', type=['jpg', 'jpeg', 'png'], key='search_image', label_visibility='collapsed')
                if uploaded is not None:
                    try:
                        st.image(uploaded, width=150)
                    except Exception:
                        pass
            with col_go:
                submitted = st.button('Tìm', type='primary', key='search_btn', width='stretch')
        else:  # hybrid
            col_q, col_img, col_go = st.columns([5, 2, 1])
            with col_q:
                st.text_input('Tìm kiếm', key='query_input', placeholder='Ví dụ: áo sơ mi trắng công sở cotton, quần jean nam, giày da nâu…', label_visibility='collapsed')
            with col_img:
                uploaded = st.file_uploader('Ảnh tham chiếu', type=['jpg', 'jpeg', 'png'], key='search_image', label_visibility='collapsed')
                if uploaded is not None:
                    try:
                        st.image(uploaded, width='stretch')
                    except Exception:
                        pass
            with col_go:
                submitted = st.button('Tìm', type='primary', key='search_btn', width='stretch')

if submitted:
    mode_now = st.session_state.get('mode_label', MODEL_OPTIONS[0])
    if mode_now == MODEL_OPTIONS[0]:  # text only
        new_image = None
    else:
        new_image = _upload_to_temp(uploaded) if uploaded is not None else st.session_state.get('last_image')
    run_search(st.session_state.get('query_input', ''), str(new_image) if new_image else None)
elif st.session_state.pop('run_search', False):
    run_search(st.session_state.get('query_input', ''), st.session_state.get('last_image'))

if not st.session_state.get('booted'):
    st.session_state['booted'] = True
    if str(st.session_state.get('query_input', '')).strip():
        run_search(st.session_state['query_input'], None)

refresh_on_filter_change()
render_recent_searches(store, on_select_key='query_input', key_prefix='recent')


def sort_results(results: list, sort_by: str) -> list:
    if sort_by == 'Tên A→Z':
        return sorted(results, key=lambda row: str(row.get('product_name', '')).lower())
    if sort_by == 'Danh mục':
        return sorted(results, key=lambda row: (str(row.get('category', '')), -float(row.get('similarity', 0.0))))
    if sort_by == 'Màu sắc':
        return sorted(results, key=lambda row: (str(row.get('color', '')), -float(row.get('similarity', 0.0))))
    return list(results)


def render_load_more(total: int, key: str) -> None:
    visible = int(st.session_state.get('visible', PAGE_SIZE))
    if visible >= total:
        return
    if st.button(f'Xem thêm ({total - visible} sản phẩm)', key=key, width='stretch'):
        st.session_state['visible'] = visible + PAGE_SIZE
        st.rerun()


if active_page == 'Khám phá':
    results = list(st.session_state.get('results') or [])
    active_filters = st.session_state.get('filters') or {}
    if results:
        render_active_filters(active_filters, key_prefix='active')
        last_query = st.session_state.get('last_query') or ''
        last_image = st.session_state.get('last_image')
        if last_image and Path(last_image).exists():
            st.image(last_image, width=140, caption='Ảnh truy vấn')
        headline = f"**{len(results)} kết quả**" + (f' cho “{last_query}”' if last_query else '')
        st.markdown(headline)
        render_query_notes(retriever, last_query)
        ordered = sort_results(results, st.session_state.get('sort_by', SORT_OPTIONS[0]))
        render_results(ordered[:int(st.session_state.get('visible', PAGE_SIZE))], store, cols=3, key_prefix='discover')
        render_load_more(len(ordered), key='discover_more')
    elif any(active_filters.values()):
        render_active_filters(active_filters, key_prefix='active')
        browsed = catalog.products(active_filters)
        st.markdown(f'**{len(browsed)} sản phẩm** khớp bộ lọc')
        render_results(browsed[:int(st.session_state.get('visible', PAGE_SIZE))], store, cols=3, show_match=False, key_prefix='browse')
        render_load_more(len(browsed), key='browse_more')
    else:
        empty_state('Tìm món đồ tiếp theo của bạn', 'Gõ mô tả, tải ảnh lên, hoặc kết hợp cả hai. Bộ lọc bên trái thu hẹp theo danh mục, màu sắc, chất liệu và phong cách.')
        render_suggestion_chips()
        for category in ('shirt', 'trouser', 'shoes', 'bag', 'dress'):
            shelf = catalog.category_shelf(category, 7)
            if len(shelf) >= 3:
                render_shelf(f'{CATEGORIES_NAMES.get(category, category)}', shelf, store, key_prefix=f'home_{category}', cols=7)

elif active_page == 'Tủ đồ':
    wardrobe_ids = store.wardrobe
    st.markdown(f'### Tủ đồ của bạn · {len(wardrobe_ids)} sản phẩm')
    if not wardrobe_ids:
        empty_state('Tủ đồ đang trống', 'Nhấn ♡ trên bất kỳ sản phẩm nào để lưu lại cho lần sau.')
    else:
        items = catalog.get_many(wardrobe_ids)
        render_results(items, store, cols=4, show_match=False, key_prefix='wardrobe')
        with st.expander('Sao lưu & khôi phục'):
            st.download_button(
                'Tải tủ đồ (.json)',
                data=store.export_json().encode('utf-8'),
                file_name=f'fashionfinder-wardrobe-{sid}.json',
                mime='application/json',
                width='stretch',
            )
            backup = st.file_uploader('Khôi phục từ tệp sao lưu', type=['json'], key='wardrobe_import')
            if backup is not None and st.button('Nhập vào tủ đồ', key='wardrobe_import_btn', width='stretch'):
                try:
                    added = store.import_json(backup.read().decode('utf-8'))
                    st.success(f'Đã thêm {added} sản phẩm vào tủ đồ.')
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if st.button('Xóa toàn bộ tủ đồ', key='wardrobe_clear'):
                store.clear_wardrobe()
                st.rerun()
    if store.saved_outfits:
        st.divider()
        st.markdown('### Set đồ đã lưu')
        for outfit in store.saved_outfits:
            with st.container(border=True):
                header_cols = st.columns([4, 1])
                with header_cols[0]:
                    st.markdown(f"**{outfit.get('name', 'Set đồ')}**")
                    st.caption(f"{len(outfit.get('items', []))} món · {str(outfit.get('created_at', ''))[:10]}")
                with header_cols[1]:
                    if st.button('Xóa', key=f"del_outfit_{outfit.get('id')}", width='stretch'):
                        store.remove_outfit(str(outfit.get('id')))
                        st.rerun()
                render_compact_row(catalog.get_many(outfit.get('items', [])), store, key_prefix=f"outfit_{outfit.get('id')}")

elif active_page == 'Phối đồ':
    st.markdown('### Phối đồ')
    wardrobe_ids = store.wardrobe
    source_ids = wardrobe_ids if wardrobe_ids else [p['id'] for p in catalog.products(limit=400)]
    labels = {}
    for product in catalog.get_many(source_ids):
        labels[product['id']] = f"{product['product_name']} · {CATEGORIES_NAMES.get(product['category'], product['category'])}"
    if not labels:
        empty_state('Chưa có sản phẩm để phối', 'Hãy lưu vài món vào tủ đồ trước.')
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            anchor_id = st.selectbox(
                'Chọn món đồ gốc',
                options=list(labels.keys()),
                format_func=lambda pid: labels.get(pid, pid),
                key='outfit_anchor',
            )
        with col_b:
            st.caption('Nguồn gợi ý: ' + ('Tủ đồ của bạn' if wardrobe_ids else 'Toàn bộ danh mục'))
        anchor = catalog.get(anchor_id)
        if anchor is not None:
            info_col, option_col = st.columns([1, 2])
            with info_col:
                if anchor.get('image_path') and Path(anchor['image_path']).exists():
                    st.image(anchor['image_path'], width='stretch')
            with option_col:
                st.markdown(f"**{anchor['product_name']}**")
                targets = catalog.outfit_targets(anchor_id)
                picked = st.pills('Phối với danh mục', options=targets, default=targets[0] if targets else None, key='outfit_target')
                outfit_mode = st.radio(
                    'Kiểu phối màu',
                    options=['tone_on_tone', 'contrast'],
                    format_func=lambda m: 'Ton-sur-ton (cùng tông)' if m == 'tone_on_tone' else 'Tương phản (nổi bật)',
                    horizontal=True,
                    key='outfit_mode_color',
                )
                per_pick = st.slider('Số gợi ý mỗi danh mục', 1, 6, 3, key='outfit_per_pick')
                save_cols = st.columns(2)
                with save_cols[0]:
                    outfit_name = st.text_input('Tên set đồ', value=f"Set {anchor['product_name'][:18]}", key='outfit_name', label_visibility='collapsed')
                with save_cols[1]:
                    if st.button('Lưu set đồ', key='outfit_save', width='stretch'):
                        pieces = [anchor_id]
                        for target in targets:
                            pieces.extend(p['id'] for p in catalog.complete_the_look(anchor_id, target, mode=outfit_mode, top_k=per_pick))
                        store.save_outfit(outfit_name, pieces)
                        st.success('Đã lưu set đồ vào tủ đồ cá nhân.')
            st.divider()
            if picked:
                st.markdown(f"**Gợi ý {CATEGORIES_NAMES.get(picked, picked).lower()} phối cùng**")
                look = catalog.complete_the_look(anchor_id, target_category=picked, mode=outfit_mode, top_k=max(4, per_pick * 2))
                render_results(look, store, cols=4, show_match=True, key_prefix=f'look_{picked}')
            st.markdown('**Cả set đồ gợi ý**')
            for target in targets:
                shelf_items = catalog.complete_the_look(anchor_id, target, mode=outfit_mode, top_k=per_pick)
                if shelf_items:
                    render_shelf(CATEGORIES_NAMES.get(target, target), shelf_items, store, key_prefix=f'set_{target}', cols=per_pick)

else:
    st.markdown('### Benchmark mô hình')
    st.caption('Màn hình nội bộ: so sánh các backbone trên cùng một truy vấn và xem bảng đánh giá offline.')
    bench_cols = st.columns([1, 1])
    with bench_cols[0]:
        bench_text = st.text_area('Mô tả truy vấn', placeholder='áo thun cotton đen thể thao', height=100, key='bench_text')
    with bench_cols[1]:
        bench_upload = st.file_uploader('Ảnh truy vấn', type=['jpg', 'jpeg', 'png'], key='bench_image')
        if bench_upload is not None:
            try:
                st.image(bench_upload, width='stretch')
            except Exception:
                pass
    if st.button('Chạy so sánh', key='bench_run', type='primary'):
        bench_image = _upload_to_temp(bench_upload) if bench_upload is not None else None
        if not bench_image and not (bench_text or '').strip():
            st.warning('Cần ít nhất một ảnh hoặc một mô tả.')
        else:
            columns = st.columns(len(BENCHMARK_COLUMNS))
            for column, (label, kind, model) in zip(columns, BENCHMARK_COLUMNS):
                with column:
                    st.markdown(f'**{label}**')
                    try:
                        if kind == 'image':
                            found = retriever.search_by_image(str(bench_image), model=model, top_k=10) if bench_image else []
                        elif kind == 'text':
                            found = retriever.search_by_text(bench_text, model=model, top_k=10) if (bench_text or '').strip() else []
                        else:
                            found = retriever.search_multimodal(img_path=str(bench_image) if bench_image else None, text=bench_text if (bench_text or '').strip() else None, alpha=float(st.session_state.get('alpha', RETRIEVAL.FUSION_ALPHA_DEFAULT)), top_k=10)
                        render_product_grid(found[:5], cols=1)
                    except Exception as exc:
                        st.error(str(exc))
    st.divider()
    metrics_csv = PATHS.REPORTS_DIR / 'metrics.csv'
    if metrics_csv.exists():
        st.markdown('**Bảng đánh giá offline**')
        table = pd.read_csv(metrics_csv).round(4)
        st.dataframe(table.sort_values(f'mAP@10', ascending=False).reset_index(drop=True), width='stretch', hide_index=True)
    else:
        st.caption('Chưa có reports/metrics.csv — chạy pipeline để sinh bảng đánh giá.')

    st.divider()
    st.markdown('**Biểu đồ so sánh**')
    bar_chart = PATHS.REPORTS_FIGURES_DIR / 'metrics_bar_chart.png'
    recall_curve = PATHS.REPORTS_FIGURES_DIR / 'recall_curve.png'
    figure_cols = st.columns(2)
    for column, (path, caption) in zip(figure_cols, ((bar_chart, 'Recall@K & mAP/F1 theo mô hình'), (recall_curve, 'Đường Recall@K'))):
        with column:
            if path.exists():
                st.image(Image.open(path), caption=caption, width='stretch')
            else:
                st.caption(f'Chưa có {path.name} — chạy `scripts/05_evaluate_all.py` để tạo.')

    if PATHS.ERROR_ANALYSIS_MD.exists():
        with st.expander('Phân tích lỗi (reports/error_analysis.md)', expanded=False):
            st.markdown(PATHS.ERROR_ANALYSIS_MD.read_text(encoding='utf-8', errors='replace'))
        error_cases = sorted((PATHS.REPORTS_DIR / 'error_cases').glob('*.png'))
        if error_cases:
            with st.expander(f'Ảnh case lỗi ({len(error_cases)})', expanded=False):
                for start in range(0, len(error_cases), 4):
                    case_cols = st.columns(4)
                    for column, case_path in zip(case_cols, error_cases[start:start + 4]):
                        with column:
                            st.image(str(case_path), caption=case_path.stem, width='stretch')

detail_id = st.session_state.pop('detail_pid', None)
if detail_id:
    known_similarity = None
    for item in (st.session_state.get('results') or []):
        if str(item.get('id')) == str(detail_id):
            known_similarity = float(item.get('similarity', 0.0))
            break
    show_product_detail(catalog, store, str(detail_id), known_similarity)

st.divider()
st.caption('FashionFinder · tìm kiếm đa phương thức cho thời trang · dữ liệu cục bộ, không gửi ảnh ra ngoài.')
