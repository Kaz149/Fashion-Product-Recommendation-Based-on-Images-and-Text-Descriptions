import sys
import tempfile
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap_env
import streamlit as st

st.set_page_config(page_title='Fashion Recommendation', page_icon='👗', layout='wide', initial_sidebar_state='expanded')
warnings.filterwarnings('ignore')
import pandas as pd
from PIL import Image
from config import CATEGORIES, DEVICE, EVALUATION, PATHS, RETRIEVAL
from src.retrieval import Retriever
from src.ui_utils import admin_sidebar_components, render_product_grid, stale_banner


@st.cache_resource(show_spinner=False)
def get_retriever():
    try:
        return Retriever(lazy=False, verbose=False)
    except Exception as e:
        st.error(f'Khởi tạo Retriever thất bại: {e}')
        return None


retriever = get_retriever()
stale_banner()
with st.sidebar:
    st.title('Fashion Recommendation System')
    st.divider()
    admin_sidebar_components()
    st.divider()
    st.caption(f'Device: `{DEVICE}` | Version: v1.0')
tab_img, tab_txt, tab_outfit = st.tabs(['🖼️  Tìm kiếm bằng ảnh', '📝  Tìm kiếm bằng văn bản', '👔  Gợi ý phối đồ'])


def _resolve_sample_image_path(name: str) -> Path | None:
    if name and name != '(None)':
        p = Path(PATHS.DATA_PROCESSED_IMAGES) / name
        if p.exists():
            return p
    return None


def _img_upload_to_temp(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _show_image(path, width, caption=''):
    try:
        st.image(str(path), width=width, caption=caption or None)
    except Exception:
        st.warning('Không đọc được ảnh.')


with tab_img:
    st.header('Tìm sản phẩm thời trang TƯƠNG ĐỒ từ ảnh')
    col_up, col_info = st.columns([1, 1])
    with col_up:
        img_upload = st.file_uploader('Tải ảnh lên (JPG/PNG)', type=['jpg', 'jpeg', 'png'], key='img_upload')
        sample_options = ['(None)'] + [f.name for f in sorted(list(Path(PATHS.DATA_PROCESSED_IMAGES).glob('*.jpg')))[:10]]
        sample_img_choice = st.radio('Chọn ảnh mẫu', options=sample_options, index=0, key='sample_img_choice')
    chosen_img_path = _img_upload_to_temp(img_upload) if img_upload is not None else _resolve_sample_image_path(sample_img_choice)
    with col_info:
        if chosen_img_path is not None:
            _show_image(chosen_img_path, 300, 'Ảnh input')
    st.divider()
    col_model, col_topk = st.columns([1, 1])
    with col_model:
        img_model_id = {'resnet': 'resnet', 'efficientnet_b0': 'effnet', 'clip_image': 'clip'}[st.selectbox('Mô hình ảnh embed', options=['resnet', 'efficientnet_b0', 'clip_image'], index=0, key='img_model')]
    with col_topk:
        img_topk = st.slider('Top-K kết quả', min_value=1, max_value=RETRIEVAL.DEFAULT_MAX_K, value=RETRIEVAL.DEFAULT_TOP_K, key='img_topk')
    if st.button('🔍 Tìm kiếm', key='img_search_btn', type='primary'):
        if retriever is None:
            st.warning('Vui lòng Rebuild trước khi tìm kiếm.')
        elif chosen_img_path is None:
            st.warning('Vui lòng tải ảnh lên hoặc chọn ảnh mẫu.')
        else:
            with st.spinner('Đang tìm kiếm tương đồng (image embedding)...'):
                try:
                    render_product_grid(retriever.search_by_image(str(chosen_img_path), model=img_model_id, top_k=img_topk), cols=5)
                except Exception as e:
                    st.error(f'Lỗi tìm kiếm ảnh: {e}')
with tab_txt:
    st.header('Tìm sản phẩm từ MÔ TẢ VĂN BẢN')
    st.markdown('💡 Gợi ý query (bấm để điền vào ô bên dưới):')
    suggestion_chips = ['Áo thun cotton đen thể thao', 'Quần jean xanh cổ điển nam', 'Giày da nâu trang trọng văn phòng', 'Túi xách da đen thanh lịch nữ', 'Váy đầm hồng polyester dự tiệc']
    chip_cols = st.columns(len(suggestion_chips))
    for ci, chip in enumerate(suggestion_chips):
        with chip_cols[ci]:
            if st.button(chip, key=f'txt_suggest_{ci}', use_container_width=True):
                st.session_state['text_query'] = chip
                st.rerun()
    text_query = st.text_area('Mô tả sản phẩm:', placeholder='ví dụ: áo sơ mi trắng công sở cotton thanh lịch', height=120, max_chars=600, key='text_query')
    col_model, col_topk = st.columns([1, 1])
    with col_model:
        txt_model_id = {'sbert': 'sbert', 'tfidf': 'tfidf', 'clip': 'clip'}[st.radio('Mô hình text embed', options=['sbert', 'tfidf', 'clip'], key='text_model')]
    with col_topk:
        txt_topk = st.slider('Top-K kết quả', min_value=1, max_value=RETRIEVAL.DEFAULT_MAX_K, value=RETRIEVAL.DEFAULT_TOP_K, key='txt_topk')
    if st.button('🔍 Tìm kiếm', key='txt_search_btn', type='primary'):
        if retriever is None:
            st.warning('Vui lòng Rebuild trước khi tìm kiếm.')
        elif not text_query or not text_query.strip():
            st.warning('Vui lòng nhập mô tả sản phẩm.')
        else:
            with st.spinner('Đang tìm kiếm tương đồng (text embedding)...'):
                try:
                    render_product_grid(retriever.search_by_text(text_query, model=txt_model_id, top_k=txt_topk), cols=5)
                except Exception as e:
                    st.error(f'Lỗi tìm kiếm văn bản: {e}')
with tab_outfit:
    st.header('GỢI Ý PHỐI ĐỒ đơn giản (Áo → Quần → Giày / Túi)')
    valid_fashion = list(CATEGORIES.VALID_FASHION)
    src_cat = st.selectbox('Category sản phẩm gốc', valid_fashion, key='src_cat')
    src_product_options = []
    src_df = None
    if retriever is not None:
        try:
            md = retriever.load_metadata()
            src_df = md[md.category == src_cat].reset_index(drop=True)
            src_product_options = list(src_df.id.astype(str) + ' | ' + src_df.product_name.astype(str))
        except Exception:
            src_product_options = []
    src_product_label = st.selectbox('Sản phẩm cụ thể', options=src_product_options, format_func=lambda s: s[:70], key='src_product', disabled=len(src_product_options) == 0)
    src_product_id = src_product_label.split(' | ')[0] if src_product_label else None
    if src_product_id is not None and src_df is not None and len(src_df) > 0:
        match = src_df[src_df.id.astype(str) == str(src_product_id)]
        if not match.empty:
            src_row = match.iloc[0]
            st.markdown('**Sản phẩm gốc đã chọn:**')
            src_card_col1, src_card_col2 = st.columns([1, 3])
            with src_card_col1:
                _show_image(src_row.get('image_path', ''), 250) if src_row.get('image_path', '') else st.write('⚠️ no image')
            with src_card_col2:
                st.markdown(f"**{src_row.get('product_name', '')}**")
                st.caption(f"Category: {src_row.get('category', '')} | Màu: {src_row.get('color', '')} | Chất liệu: {src_row.get('material', '')} | Phong cách: {src_row.get('style', '')}")
                st.caption(f'ID: {src_product_id}')
                with st.expander('Mô tả chi tiết'):
                    st.write(str(src_row.get('description', '')))
    st.divider()
    outfit_targets = list(CATEGORIES.OUTFIT_MAP.get(src_cat, valid_fashion))
    tgt_cat = st.radio('Đích (category muốn phối):', options=outfit_targets or valid_fashion, index=0, key='tgt_cat', horizontal=True)
    outfit_topk = st.slider('Top-K gợi ý', min_value=1, max_value=RETRIEVAL.DEFAULT_MAX_K, value=RETRIEVAL.DEFAULT_TOP_K, key='outfit_topk')
    if st.button('🔀 Gợi ý phối đồ', key='outfit_btn', type='primary'):
        if retriever is None:
            st.warning('Vui lòng Rebuild trước khi tìm kiếm.')
        elif src_product_id is None:
            st.warning('Vui lòng chọn sản phẩm gốc.')
        else:
            with st.spinner('Đang gợi ý phối đồ (SBERT attribute matching)...'):
                try:
                    render_product_grid(retriever.outfit_match(src_product_id, target_category=tgt_cat, top_k=outfit_topk), cols=5, show_explanation=True)
                except Exception as e:
                    st.error(f'Lỗi gợi ý phối đồ: {e}')


def _cmp_helper(results, label, limit=5):
    st.subheader(label)
    if not results:
        st.info('Không có kết quả.')
        return
    render_product_grid(results[:limit], cols=1, show_explanation=False)


st.divider()
st.caption('Fashion Recommendation System')
