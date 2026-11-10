
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar('T')

MAX_PAGE_SIZE = 24
PAGE_SIZE_OPTIONS = (12, 24)
DEFAULT_PAGE_SIZE = 12


def normalise_page_size(page_size: int) -> int:
    size = int(page_size)
    if size < 1:
        return DEFAULT_PAGE_SIZE
    return min(size, MAX_PAGE_SIZE)


def total_pages(total_items: int, page_size: int) -> int:
    size = normalise_page_size(page_size)
    return max(1, -(-max(0, int(total_items)) // size))


def clamp_page(page: int, total_items: int, page_size: int) -> int:
    pages = total_pages(total_items, page_size)
    try:
        current = int(page)
    except (TypeError, ValueError):
        current = 1
    return min(max(1, current), pages)


def page_bounds(page: int, total_items: int, page_size: int) -> tuple[int, int]:
    size = normalise_page_size(page_size)
    current = clamp_page(page, total_items, page_size)
    start = (current - 1) * size
    return start, min(start + size, max(0, int(total_items)))


def page_slice(items: Sequence[T], page: int, page_size: int) -> list[T]:
    start, end = page_bounds(page, len(items), page_size)
    return list(items[start:end])
