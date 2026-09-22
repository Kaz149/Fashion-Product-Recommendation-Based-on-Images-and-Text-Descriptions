from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from config import PATHS

def load_image_safe(img_path: str | Path, size: tuple | None=None, thumb: tuple=(600, 600)) -> Image.Image | None:
    try:
        img = Image.open(img_path).convert('RGB')
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
            return img
        img.thumbnail(thumb, Image.LANCZOS)
        return img
    except Exception:
        return None

def render_product_grid(results: List[Dict[str, Any]], cols: int=5, caption_template: str='{product_name}\n{category} | {color} | sim={similarity:.3f}', show_explanation: bool=False) -> None:
    if not results:
        st.info('Không có kết quả nào.')
        return
    for i in range(0, len(results), cols):
        batch = results[i:i + cols]
        row_cols = st.columns(cols)
        for j, result in enumerate(batch):
            with row_cols[j]:
                img_path = str(result.get('image_path', ''))
                shown = False
                if img_path and Path(img_path).exists():
                    try:
                        st.image(img_path, use_container_width=True)
                        shown = True
                    except Exception:
                        shown = False
                if not shown:
                    img = load_image_safe(result.get('image_path', ''))
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.write('⚠️ no image')
                safe_result = {'product_name': str(result.get('product_name', '')), 'category': str(result.get('category', '')), 'color': str(result.get('color', '')), 'material': str(result.get('material', '')), 'style': str(result.get('style', '')), 'similarity': float(result.get('similarity', 0.0)), 'id': str(result.get('id', '')), 'description': str(result.get('description', ''))}
                try:
                    caption = caption_template.format(**safe_result)
                except Exception:
                    caption = f"{safe_result['product_name']} | sim={safe_result['similarity']:.3f}"
                st.caption(caption)
                if show_explanation and 'explanation' in result and (len(result['explanation']) > 0):
                    st.caption('💡 ' + '; '.join(result['explanation']))

def stale_banner() -> None:
    from src.dataset_adaptor import is_embedding_stale
    manifest_path = PATHS.EMB_MANIFEST
    metadata_path = PATHS.DATA_PROCESSED_METADATA
    images_path = PATHS.DATA_PROCESSED_IMAGES
    if not manifest_path.exists():
        st.error('❌ Chưa có embedding nào được trích xuất. Chạy Rebuild Toàn Bộ.')
        return
    if is_embedding_stale(manifest_path, metadata_path, images_path):
        st.warning('⚠️ Embedding đã cũ (dataset thay đổi)! Vui lòng nhấn nút **Rebuild Toàn Bộ** ở thanh sidebar Admin bên trái hoặc chạy `python scripts/rebuild_all.py` ở Terminal để cập nhật.')

def admin_sidebar_components() -> None:
    st.sidebar.markdown('**Rebuild dữ liệu**')
    cols = st.sidebar.columns(2)
    with cols[0]:
        btn_rebuild = st.button('🔄 Rebuild', key='admin_rebuild', use_container_width=True)
    with cols[1]:
        force_flag = st.checkbox('Force', value=False, key='admin_force')
    if btn_rebuild:
        cmd = [sys.executable, 'scripts/rebuild_all.py']
        if force_flag:
            cmd.append('--force')
        with st.sidebar:
            with st.spinner('Đang rebuild toàn bộ pipeline...'):
                try:
                    result = subprocess.run(cmd, cwd=str(PATHS.PROJECT_ROOT), capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success('Rebuild thành công. Refresh trang để tải dữ liệu mới!')
                    else:
                        st.error('Lỗi rebuild. Xem terminal logs.')
                        if result.stderr:
                            with st.expander('Chi tiết lỗi'):
                                st.code(result.stderr[-2000:])
                except Exception as e:
                    st.error(f'Lỗi khởi chạy rebuild: {e}')
    st.sidebar.divider()
