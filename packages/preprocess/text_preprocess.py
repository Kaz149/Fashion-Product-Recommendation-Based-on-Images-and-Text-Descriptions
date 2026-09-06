import re
import string
from typing import List, Optional
import yaml


def load_config(config_path: str = "pipeline/config/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "you", "i", "me", "my",
    "we", "our", "they", "them", "their", "what", "which", "who", "whom",
    "these", "those", "am", "or", "but", "if", "then", "else", "so",
    "than", "too", "very", "can", "just", "should", "now", "also",
}


def lowercase(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


def remove_special_chars(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
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
    return " ".join(t for t in tokens if t not in stopwords)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def clean_text(
    text: str,
    remove_stop: bool = True,
    stopwords: Optional[set] = None,
) -> str:
    text = lowercase(text)
    text = remove_special_chars(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    if remove_stop:
        text = remove_stopwords(text, stopwords=stopwords)
    return text


def clean_text_batch(
    texts: List[str],
    remove_stop: bool = True,
    stopwords: Optional[set] = None,
) -> List[str]:
    return [clean_text(t, remove_stop=remove_stop, stopwords=stopwords) for t in texts]


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
