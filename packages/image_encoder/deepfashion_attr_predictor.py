# -*- coding: utf-8 -*-
"""

References:
  https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
  Reference GitHub: https://github.com/khanhnamle1994/fashion-recommendation
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image

_FASHION_COLOR_CENTROIDS_HSV = [

    ("white",   None,          (0.00, 0.25), (0.90, 1.00), 90),
    ("black",   None,          (0.00, 0.30), (0.00, 0.12), 95),
    ("gray",    None,          (0.00, 0.20), (0.12, 0.55), 85),
    ("beige",   (40, 80),      (0.10, 0.30), (0.85, 1.00), 28),  
    ("beige",   None,          (0.00, 0.06), (0.85, 1.00), 25), 
    ("khaki",   (40, 70),      (0.30, 0.70), (0.55, 0.80), 60),
    ("navy",    (200, 250),    (0.35, 1.00), (0.12, 0.35), 50),
    ("blue",    (190, 250),    (0.30, 1.00), (0.35, 1.00), 45),
    ("green",   (90, 165),     (0.25, 1.00), (0.20, 1.00), 40),
    ("brown",   (15, 45),      (0.40, 1.00), (0.20, 0.55), 42),
    ("orange",  (15, 45),      (0.55, 1.00), (0.55, 1.00), 40),
    ("yellow",  (45, 75),      (0.45, 1.00), (0.70, 1.00), 40),
    ("red",     (0, 15),       (0.40, 1.00), (0.35, 1.00), 35),
    ("red",     (330, 360),    (0.40, 1.00), (0.35, 1.00), 35),  
    ("pink",    (300, 355),    (0.20, 1.00), (0.60, 1.00), 36),
    ("pink",    (330, 350),    (0.15, 1.00), (0.55, 1.00), 35),  
    ("purple",  (255, 295),    (0.25, 1.00), (0.25, 1.00), 25),
]

_VALID_COLOR_NAMES = sorted(set(r[0] for r in _FASHION_COLOR_CENTROIDS_HSV))


def _rgb_to_hsv_flat(flat_rgb):
    
    from colorsys import rgb_to_hsv
    f = flat_rgb.astype(np.float32) / 255.0
    out = np.zeros_like(f)
    for i in range(f.shape[0]):
        h, s, v = rgb_to_hsv(f[i, 0], f[i, 1], f[i, 2])
        out[i, 0] = h * 360.0   # dùng HSV vì vượt trội hơn RGB trong computer vision
        out[i, 1] = s
        out[i, 2] = v
    return out


def _kmeans(X, K, n_iter=10, rng=None):
    rng = rng or np.random.RandomState(42)
    N = X.shape[0]
    if N < K:
        return X[:N], np.zeros(N, dtype=int)
    idx0 = rng.choice(N, K, replace=False)
    cents = X[idx0].astype(np.float32).copy()
    Z = None
    for _ in range(n_iter):
        dists = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        Z = np.argmin(dists, axis=1)
        for k in range(K):
            mask = Z == k
            if mask.any():
                cents[k] = X[mask].mean(axis=0)
    return cents, Z


class DeepFashionColorPredictor:
    
    VALID_COLORS = _VALID_COLOR_NAMES

    def __init__(self, max_pixels=5000, k=3):
        self.max_pixels = max_pixels
        self.k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_color(self, image_path: str) -> str:
        try:
            hsv_centroids, weights = self._dominant_hsv_centroids(image_path)
        except Exception:
            return ""
        best_label = None
        best_score = -1.0
        for (hh, ss, vv), w in zip(hsv_centroids, weights):
            # Kiểm tra từng ngưỡng màu 14 quy tắc trong bảng mã màu HSV
            for name, h_range, s_range, v_range, priority in _FASHION_COLOR_CENTROIDS_HSV:
                # Nếu ngưỡng có khác màu xám/đen/trắng ko hue -> check hue
                if h_range is not None:
                    hlo, hhi = h_range
                    if not (hlo <= hh <= hhi):
                        continue
                # Kiểm tra saturation 
                slo, shi = s_range
                if not (slo <= ss <= shi):
                    continue
                # Kiểm tra value độ sáng 
                vlo, vhi = v_range
                if not (vlo <= vv <= vhi):
                    continue
                # Điểm: (weight tỷ lệ % pixel của centroid + 0.01 offset tránh =0) * priority
                score = float(priority) * float(w + 0.01)
                if score > best_score:
                    best_score = score
                    best_label = name
        if best_label is None:
            _, _, vv = hsv_centroids[0]
            if vv > 0.9:
                return "white"
            if vv < 0.12:
                return "black"
            return "gray"
        return best_label

    def predict_color_batch(self, paths):
        return [self.predict_color(p) for p in paths]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _dominant_hsv_centroids(self, image_path):

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        img = Image.open(image_path).convert("RGB")
        # Resize xuống max 256px để giảm độ phức tạp K-means
        W, H = img.size
        scale = min(1.0, 256.0 / max(W, H))
        if scale < 1.0:
            img = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.LANCZOS)
        arr = np.asarray(img)
        flat = arr.reshape(-1, 3)
        # Nếu ảnh > 5000 pixel thì random subsample 5000 
        if flat.shape[0] > self.max_pixels:
            rng = np.random.RandomState(42)
            sel = rng.choice(flat.shape[0], self.max_pixels, replace=False)
            flat = flat[sel]
        # RGB -> HSV cho từng pixel 
        hsv_flat = _rgb_to_hsv_flat(flat)
        #  K-means tìm màu chủ đạo, 10 vòng lặp đủ hội tụ
        cents_hsv, Z = _kmeans(hsv_flat, K=self.k, n_iter=10)
        N = Z.shape[0]
        counts = np.array([(Z == k).sum() for k in range(cents_hsv.shape[0])], dtype=np.float32)
        weights = counts / counts.sum()
        # Sắp xếp centroid DESC theo weight 
        order = np.argsort(-weights)
        return cents_hsv[order], weights[order]


def predict_dominant_color_rgb(image_path: str, n_colors: int = 3):

    dp = DeepFashionColorPredictor()
    hsv_c, w = dp._dominant_hsv_centroids(image_path)
    from colorsys import hsv_to_rgb
    rgbs = []
    for (h, s, v), _ in zip(hsv_c, w):
        r, g, b = hsv_to_rgb(h / 360.0, s, v)
        rgbs.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return rgbs
