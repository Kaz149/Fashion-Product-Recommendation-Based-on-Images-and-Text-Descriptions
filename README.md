# Multimodal Fashion Recommendation System

Tìm kiếm và phối đồ thời trang bằng **hình ảnh**, **mô tả văn bản** (tiếng Việt / tiếng Anh) hoặc **kết hợp cả hai**.
Repo này là bản hợp nhất: giao diện cửa hàng (FashionFinder) + toàn bộ pipeline huấn luyện/đánh giá của bản
`Fashion-Product-Recommendation-Based-on-Images-and-Text-Descriptions`.

## Repository Structure

```
project-root/
├── app.py                      # Streamlit storefront (Khám phá / Tủ đồ / Phối đồ / Benchmark)
├── config.py                   # PATHS, model config, ngưỡng retrieval/evaluation
├── run_all.bat                 # 1-click: venv + pip + tải weights + rebuild pipeline + streamlit
├── _bootstrap_env.py           # Nạp .venv / sys.path cho mọi script
├── src/
│   ├── dataset_adaptor.py      # DatasetLoader (format A/B/C) + MetadataGenerator song ngữ
│   ├── image_embedder.py       # ResNet50 / EfficientNet-B0 / CLIP image
│   ├── text_embedder.py        # TF-IDF / SBERT / CLIP text
│   ├── retrieval.py            # Retriever: image / text / multimodal / outfit_match (+ filters)
│   ├── catalog.py              # Catalog: sản phẩm, facet, trang chi tiết, complete_the_look
│   ├── filters.py              # Facet mask dùng chung cho UI và retrieval
│   ├── metrics.py              # P@k / R@k / AP / mAP / F1 + evaluate_method()
│   ├── pagination.py           # Chia trang kết quả
│   ├── user_store.py           # Tủ đồ + lịch sử tìm kiếm theo phiên
│   ├── ui_utils.py             # Widget Streamlit dùng chung
│   ├── services/app_services.py# Truy vấn + session state (tách khỏi view)
│   └── views/                  # View Khám phá / thanh tìm kiếm
├── scripts/
│   ├── download_weights.py         # Tải & cache SBERT + CLIP (chạy 1 lần, cần mạng)
│   ├── 01_prepare_dataset.py       # Chuẩn hoá dataset -> metadata.csv + ảnh 224x224
│   ├── 02_extract_all_embeddings.py# Trích xuất 7 loại embedding
│   ├── 03_build_embedding_manifest.py
│   ├── 04_build_ground_truth.py
│   ├── 05_evaluate_all.py          # metrics.csv + biểu đồ PNG
│   ├── 06_error_analysis.py        # error_analysis.md + error_cases/*.png
│   ├── rebuild_all.py              # Chạy toàn bộ pipeline (--only-steps)
│   └── rebuild_tfidf.py            # Chỉ dựng lại TF-IDF (giữ nguyên embedding khác, có backup)
├── tests/                      # pytest: retrieval, product layer, smoke
├── data/                       # raw / processed / embeddings / results / user_state
├── models/                     # TF-IDF vectorizer, checkpoint
└── reports/                    # metrics.csv, error_analysis.md, figures/, error_cases/
```

## Demo

### 0. Yêu cầu trước khi chạy

| Thành phần | Yêu cầu | Ghi chú |
|---|---|---|
| Python | 3.10+ (khuyến nghị 3.11–3.13) | `python --version` |
| pip | đi kèm Python | `python -m pip --version` |
| RAM | ≥ 8 GB | CLIP + ResNet50 + EfficientNet-B0 nạp cùng lúc |
| Đĩa | ≥ 3 GB trống | ~1 GB pretrained weights + embedding `.npy` |
| GPU | không bắt buộc | tự chạy CPU nếu không có CUDA |
| Internet | **cần ở lần chạy đầu** | để tải pretrained weights (bước 2.5) |

Dữ liệu đầu vào: đặt ảnh vào `data/raw/images` (mặc định, format `C` = flat folder).
Nếu chưa có ảnh, chạy `python scripts/01_prepare_dataset.py` trước để kiểm tra format.

### 1. Cài dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` gồm: `torch`, `torchvision`, `open-clip-torch`, `sentence-transformers`,
`scikit-learn`, `streamlit`, `pandas`, `numpy`, `Pillow`, `opencv-python-headless`,
`matplotlib`, `seaborn`, `tqdm`, `PyYAML`, `joblib`.

Riêng test cần thêm `pytest` (không nằm trong `requirements.txt`):

```bash
pip install pytest
```

Kiểm tra nhanh cài đặt:

```bash
python -c "import torch, sentence_transformers, open_clip, streamlit, cv2; print('deps ok')"
```

### 2. Tải pretrained weights (bắt buộc với SBERT / CLIP)

Repo bật sẵn chế độ offline của Hugging Face (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
trong `_bootstrap_env.py`) để pipeline eval không tự gọi mạng. Vì vậy **lần chạy đầu phải tắt
offline trong đúng cửa sổ terminal đó**, nếu không sẽ gặp lỗi:

> `Không tìm được kết quả: We couldn't connect to 'https://huggingface.co' to load the files,
> and couldn't find them in the cached files.`

**Windows (PowerShell):**

```powershell
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"
python scripts/download_weights.py
```

Hoặc chạy từng lệnh thủ công:

```powershell
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"
python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2'); print('sbert ok')"
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); print('clip ok')"
```

**macOS / Linux:**

```bash
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python scripts/download_weights.py
```

Hoặc chạy từng lệnh thủ công:

```bash
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2'); print('sbert ok')"
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); print('clip ok')"
```

Chi tiết tải: SBERT `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB), CLIP
`ViT-B-32 laion2b_s34b_b79k` (~600 MB), ResNet50 / EfficientNet-B0 (~100 MB) qua `torchvision`.

Cache mặc định là `%USERPROFILE%\.cache\huggingface` (Windows) hoặc `~/.cache/huggingface`.
Nếu muốn để cache ở ổ khác, đặt trước khi tải:

```powershell
$env:HF_HOME = "D:\hf-cache"
```

Sau khi tải xong, các lần chạy sau **không cần Internet** và có thể bật lại offline.

**Không có Internet?** Dùng `tfidf` thay cho `sbert`: TF-IDF đọc `models/tfidf_vectorizer.pkl`
có sẵn, không cần tải gì. Trong app: sidebar → **Tuỳ chọn mô hình** → **Mô hình văn bản** = `tfidf`.

### 3. Chuẩn bị dữ liệu + embedding + ground truth + đánh giá

```bash
python scripts/rebuild_all.py --dataset-format C --dataset-path data/raw/images
```

Chạy nhanh từng phần (đúng thứ tự):

```bash
python scripts/01_prepare_dataset.py --dataset-format C --dataset-path data/raw/images
python scripts/02_extract_all_embeddings.py --type all --skip-existing
python scripts/03_build_embedding_manifest.py
python scripts/04_build_ground_truth.py
python scripts/05_evaluate_all.py
python scripts/06_error_analysis.py
```

Chỉ dựng lại TF-IDF khi metadata đổi từ vựng nhưng không muốn trích xuất lại các embedding còn lại:

```bash
python scripts/rebuild_tfidf.py
```

### 4. Chạy web app

```bash
streamlit run app.py
```

Mở http://localhost:8501. Tắt server bằng `Ctrl+C`.

Hoặc bấm `run_all.bat` trên Windows để làm toàn bộ từ đầu (venv + pip + tải weights + rebuild + Streamlit).

## Cài đặt & khởi chạy với 1 script

Nếu không muốn tự chạy từng bước, chỉ cần:

```bash
python scripts/download_weights.py     # tải SBERT + CLIP (1 lần, cần mạng)
python scripts/rebuild_all.py --dataset-format C --dataset-path data/raw/images
streamlit run app.py
```

Script `download_weights.py` tự kiểm tra cache và chỉ tải phần còn thiếu. Các cờ:
`--check` (chỉ báo trạng thái, không tải), `--force` (tải lại dù đã cache).

## Giao diện

- **Khám phá**: tìm bằng văn bản, ảnh, hoặc cả hai (alpha điều chỉnh trọng số), lọc theo danh mục / màu / chất liệu / phong cách, sắp xếp, gợi ý query, chọn **ảnh mẫu** trong danh mục, hiển thị tự sửa query và màu gần giống.
- **Tủ đồ**: lưu sản phẩm, sao lưu / khôi phục JSON, lưu set đồ.
- **Phối đồ**: gợi ý ton-sur-ton hoặc tương phản cho từng danh mục, lưu thành set.
- **Benchmark** (bật *Chế độ quản trị*): so sánh CLIP Image / ResNet50 / EfficientNet-B0 / SBERT / CLIP Hybrid trên cùng input, bảng `metrics.csv`, biểu đồ Recall@K, `error_analysis.md` và ảnh các case lỗi.
- **Chế độ quản trị**: `Rebuild` (toàn bộ pipeline) và `Rebuild TF-IDF` (chỉ TF-IDF).

## Tests

```bash
python -m pytest tests -q
```

## Ghi chú

- Đường dẫn ảnh trong `metadata.csv` / `index_mapping.pkl` được resolve lại theo thư mục checkout hiện tại, nên copy/move project sang máy khác vẫn hiển thị ảnh đúng.
- Nếu chuyển sang bộ từ vựng TF-IDF mới, phải chạy lại `rebuild_tfidf.py` rồi restart Streamlit.
- `SBERT` / `CLIP` cần tải pretrained weights lần đầu (mạng + cache HuggingFace/OpenCLIP).
