import re
import string
import unicodedata
from typing import List, Optional, Dict
import yaml


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "you", "i", "me", "my",
    "we", "our", "they", "them", "their", "what", "which", "who", "whom",
    "these", "those", "am", "or", "but", "if", "then", "else", "so",
    "than", "too", "very", "can", "just", "should", "now", "also",
}


VI_STOPWORDS = {
    "là", "và", "của", "có", "trong", "với", "những", "nhất", "này", "đó",
    "các", "một", "như", "nào", "mà", "rất", "thì", "được", "vì", "theo",
    "hoặc", "nếu", "sẽ", "đã", "từ", "đến", "khi", "đang", "vẫn", "hơn",
    "nhiều", "ít", "cũng", "chỉ", "tôi", "bạn", "anh", "chị", "em", "ông",
    "bà", "họ", "chúng", "tao", "mày", "gì", "đâu", "sao", "bao", "nhiêu",
    "thứ", "khoảng", "lên", "xuống", "trong", "ngoài", "trên", "dưới",
    "bên", "ngay", "luôn", "quá", "thật", "biết", "đi", "lại", "trở",
    "nên", "cho", "để", "về", "ra", "vào", "qua", "từng", "cùng", "nhau",
}


DEFAULT_STOPWORDS = EN_STOPWORDS | VI_STOPWORDS


CATEGORY_VI_TO_EN: Dict[str, str] = {
    "áo": "shirt", "áo sơ mi": "shirt", "sơ mi": "shirt", "áo thun": "shirt",
    "shirt": "shirt", "tshirt": "shirt", "t-shirt": "shirt", "tee": "shirt",
    "blouse": "shirt", "top": "shirt", "sweater": "shirt", "hoodie": "shirt",
    "áo khoác": "jacket", "khoác": "jacket", "ao khoac": "jacket",
    "jacket": "jacket", "coat": "jacket", "bomber": "jacket", "blazer": "jacket",
    "quần": "pants", "quần jean": "pants", "quần tây": "pants", "quần kaki": "pants",
    "jean": "pants", "quan": "pants", "quan jean": "pants",
    "pants": "pants", "jeans": "pants", "trousers": "pants", "chinos": "pants", "khakis": "pants", "slacks": "pants",
    "giày": "shoes", "giày da": "shoes", "giày thể thao": "shoes",
    "sneaker": "shoes", "sneakers": "shoes", "dep": "shoes", "dép": "shoes",
    "shoes": "shoes", "shoe": "shoes", "boots": "shoes", "sneaker": "shoes", "sandals": "shoes", "oxford": "shoes",
    "váy": "skirt", "váy liền": "dress", "đầm": "dress",
    "vay": "skirt", "dam": "dress", "vay lien": "dress",
    "skirt": "skirt", "dress": "dress", "gown": "dress",
    "phụ kiện": "accessory", "túi": "accessory", "balo": "accessory",
    "khăn": "accessory", "mũ": "accessory", "trang sức": "accessory",
    "accessory": "accessory", "accessories": "accessory", "bag": "accessory", "backpack": "accessory",
    "hat": "accessory", "cap": "accessory", "scarf": "accessory", "jewelry": "accessory",
    "khác": "other", "không xác định": "other", "other": "other",
}


COLOR_VI_TO_EN: Dict[str, str] = {
    "trắng": "white", "white": "white", "trang": "white",
    "đen": "black", "black": "black", "den": "black",
    "xanh": "blue", "lam": "blue", "blue": "blue",
    "xanh dương": "blue", "xanh da trời": "blue",
    "xanh lá": "green", "green": "green",
    "đỏ": "red", "red": "red", "do": "red",
    "vàng": "yellow", "yellow": "yellow", "vang": "yellow",
    "hồng": "pink", "pink": "pink", "hong": "pink",
    "tím": "purple", "purple": "purple", "tim": "purple",
    "cam": "orange", "orange": "orange",
    "nâu": "brown", "brown": "brown", "nau": "brown",
    "be": "beige", "beige": "beige", "bé": "beige",
    "xám": "gray", "gray": "gray", "grey": "gray", "xam": "gray",
    "kaki": "khaki", "khaki": "khaki",
    "tây ban nha": "navy", "navy": "navy", "navy blue": "navy",
}


STYLE_VI_TO_EN: Dict[str, str] = {
    "thường": "casual", "casual": "casual", "du lịch": "casual",
    "công sở": "formal", "formal": "formal", "tiệc": "formal",
    "dễ thương": "smart", "thanh lịch": "smart", "smart": "smart",
    "thể thao": "sporty", "sport": "sporty", "sporty": "sporty",
    "học đường": "casual", "basic": "casual",
}


MATERIAL_VI_TO_EN: Dict[str, str] = {
    "cotton": "cotton", "bông": "cotton", "thoải mát": "cotton",
    "jean": "denim", "denim": "denim", "bạt": "denim",
    "da": "leather", "leather": "leather", "da thật": "leather",
    "len": "wool", "wool": "wool", "merino": "wool",
    "lụa": "silk", "silk": "silk", "tơ": "silk",
    "poly": "polyester", "polyester": "polyester", "vải tổng hợp": "polyester",
    "voan": "chiffon", "chiffon": "chiffon",
    "kaki": "cotton twill", "thun": "cotton blend",
}


def lowercase(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


def strip_accents(text: str) -> str:

    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_acc.replace("đ", "d").replace("Đ", "D")


def remove_special_chars(text: str, keep_vietnamese_accents: bool = True) -> str:
    if not text:
        return ""
    # xóa URL 
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # xóa  HTML (<tag>..</tag>)
    text = re.sub(r"<[^>]+>", " ", text)
    if keep_vietnamese_accents:
        
        text = re.sub(r"[^A-Za-z0-9À-ỹĐđ\s]", " ", text)
    else:
       
        text = re.sub(r"[^\w\s]", " ", text)
    return text


def remove_extra_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str) -> str:
    if not text:
        return ""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(
    text: str,
    stopwords: Optional[set] = None,
) -> str:
    if not text:
        return ""
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    tokens = text.split()
    no_stop = [t for t in tokens if t not in stopwords]
    return " ".join(no_stop)


def map_category_vi_to_en(category: str) -> str:

    if not isinstance(category, str) or not category.strip():
        return "other"
    key = category.strip().lower()
    if key in CATEGORY_VI_TO_EN:
        return CATEGORY_VI_TO_EN[key]
    key_no_acc = strip_accents(key)
    if key_no_acc in CATEGORY_VI_TO_EN:
        return CATEGORY_VI_TO_EN[key_no_acc]
    for phrase, val in CATEGORY_VI_TO_EN.items():
        if phrase in key or phrase in key_no_acc:
            return val
    return key if len(key) <= 15 else "other"


def map_color_vi_to_en(color: str) -> str:
    if not isinstance(color, str) or not color.strip():
        return ""
    key = color.strip().lower()
    if key in COLOR_VI_TO_EN:
        return COLOR_VI_TO_EN[key]
    key_no_acc = strip_accents(key)
    if key_no_acc in COLOR_VI_TO_EN:
        return COLOR_VI_TO_EN[key_no_acc]
    for phrase, val in COLOR_VI_TO_EN.items():
        if phrase in key or phrase in key_no_acc:
            return val
    return key


def map_style_vi_to_en(style: str) -> str:
    if not isinstance(style, str) or not style.strip():
        return "casual"
    key = style.strip().lower()
    if key in STYLE_VI_TO_EN:
        return STYLE_VI_TO_EN[key]
    key_no_acc = strip_accents(key)
    if key_no_acc in STYLE_VI_TO_EN:
        return STYLE_VI_TO_EN[key_no_acc]
    for phrase, val in STYLE_VI_TO_EN.items():
        if phrase in key or phrase in key_no_acc:
            return val
    return "casual"


def map_material_vi_to_en(material: str) -> str:
    if not isinstance(material, str) or not material.strip():
        return ""
    key = material.strip().lower()
    if key in MATERIAL_VI_TO_EN:
        return MATERIAL_VI_TO_EN[key]
    key_no_acc = strip_accents(key)
    if key_no_acc in MATERIAL_VI_TO_EN:
        return MATERIAL_VI_TO_EN[key_no_acc]
    for phrase, val in MATERIAL_VI_TO_EN.items():
        if phrase in key or phrase in key_no_acc:
            return val
    return key


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def clean_text(
    text: str,
    remove_stop: bool = True,
    stopwords: Optional[set] = None,
    keep_vietnamese_accents: bool = True,
) -> str:
    text = lowercase(text)
    text = remove_special_chars(text, keep_vietnamese_accents=keep_vietnamese_accents)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    if remove_stop:
        text = remove_stopwords(text, stopwords=stopwords)
    return text


def clean_text_batch(
    texts: List[str],
    remove_stop: bool = True,
    stopwords: Optional[set] = None,
    keep_vietnamese_accents: bool = True,
) -> List[str]:
    return [
        clean_text(t, remove_stop=remove_stop, stopwords=stopwords, keep_vietnamese_accents=keep_vietnamese_accents)
        for t in texts
    ]


def build_description(
    name: str = "",
    category: str = "",
    color: str = "",
    material: str = "",
    style: str = "",
    description: str = "",
    use_fields: bool = True,
) -> str:
    if not use_fields:
        return clean_text(description)
    parts = [p for p in [name, category, color, material, style, description] if isinstance(p, str) and p.strip()]
    return clean_text(" ".join(parts))
