# Fashion Product Recommendation System

## Repository Structure

```
project-root/
├── src/                      # Source code modules
│   ├── dataset_adaptor.py    # DatasetLoader 3 format, MetadataGenerator song ngữ
│   ├── image_embedder.py     # BaseImageEmbedder + 3 lớp (ResNet50/EffNetB0/CLIP)
│   ├── text_embedder.py      # BaseTextEmbedder + 3 lớp (TF-IDF/SBERT/CLIP)
│   ├── retrieval.py          # Retriever class 4 modes + outfit match 
│   ├── metrics.py            # P@k/R@k/AP/mAP/F1 + evaluate_method()
│   └── ui_utils.py           # Streamlit helpers 
│
├─ requirements.txt
└─ README.md
```
## Demo

# 1. Install dependencies 
pip install -r requirements.txt

# 2. Prepare data + embeddings 
python scripts/01_prepare_dataset.py --force
python scripts/02_extract_all_embeddings.py
python scripts/03_build_embedding_manifest.py

# 3. Launch the web app
streamlit run app.py
