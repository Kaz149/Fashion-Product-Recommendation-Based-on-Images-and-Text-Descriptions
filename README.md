# Fashion Product Recommendation System

## Repository Structure

```
project-root/
├─ packages/                # Core Python packages
│  ├─ preprocess/          # Module 1: Data Preprocessing
│  │  ├─ image_preprocess.py   # Resize, normalize images
│  │  ├─ text_preprocess.py    # Clean, tokenize text descriptions
│  │  └─ dataset_loader.py     # Load dataset, split train/val/test
│  │
│  ├─ image_encoder/       # Module 2: Image Embedding
│  │  ├─ resnet_encoder.py     # ResNet50 
│  │  ├─ clip_image_encoder.py  # CLIP ViT image branch 
│  │  └─deepfashion_attr_predictor.py   # predict color from pixel
│  │
│  ├─ text_encoder/        # Module 3: Text Embedding
│  │  
│  ├─ fusion/              # Module 4: Multimodal Fusion
│  │
│  ├─ retrieval/           # Module 5: Similarity Search
│  │
│  ├─ outfit/              # Module 6: Outfit Coordination
│  │
│  └─ evaluation/          # Module 7: Evaluation & Comparison
│
├─ pipeline/               # End-to-end pipeline
│
├─ webapp/                 # Module 8: Web Demo (Streamlit)
│
├─ data/
│  ├─ raw/                    # Raw dataset (images + metadata CSV)
│  └─ embeddings/             # Saved product embeddings
│
├─ requirements.txt
└─ README.md
```

