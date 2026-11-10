from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MAX_RECENT_SEARCHES = 12
MAX_RECENTLY_VIEWED = 24
MAX_SAVED_OUTFITS = 30
MAX_WARDROBE_ITEMS = 200
STATE_VERSION = 1


def new_session_id() -> str:
    return secrets.token_hex(6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _as_id(product_id: Any) -> str:
    return str(product_id).strip()


def _dedupe(ids: Iterable[Any], limit: int) -> List[str]:
    seen: List[str] = []
    for raw in ids:
        pid = _as_id(raw)
        if pid and pid not in seen:
            seen.append(pid)
    return seen[:limit]


def empty_state(session_id: str) -> Dict[str, Any]:
    return {
        'version': STATE_VERSION,
        'session_id': session_id,
        'created_at': _now(),
        'updated_at': _now(),
        'wardrobe': [],
        'recently_viewed': [],
        'recent_searches': [],
        'saved_outfits': [],
    }

class UserStore:

    def __init__(self, path: Path, data: Dict[str, Any]):
        self.path = Path(path)
        self.data = data

    @classmethod
    def load(cls, root: Path, session_id: Optional[str] = None) -> 'UserStore':
        session_id = (session_id or '').strip() or new_session_id()
        root = Path(root)
        path = root / f'{session_id}.json'
        data: Dict[str, Any] = empty_state(session_id)
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(loaded, dict):
                    data.update({k: v for k, v in loaded.items() if v is not None})
            except Exception:
                data = empty_state(session_id)
        data['session_id'] = session_id
        data.setdefault('version', STATE_VERSION)
        for key in ('wardrobe', 'recently_viewed', 'recent_searches', 'saved_outfits'):
            if not isinstance(data.get(key), list):
                data[key] = []
        return cls(path, data)

    @property
    def session_id(self) -> str:
        return str(self.data.get('session_id', ''))

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data['updated_at'] = _now()
            payload = json.dumps(self.data, ensure_ascii=False, indent=2)
            fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(payload)
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        except Exception:
            pass

    # ---------------------------------------------------------------- wardrobe
    @property
    def wardrobe(self) -> List[str]:
        return list(self.data.get('wardrobe', []))

    def is_favorite(self, product_id: Any) -> bool:
        return _as_id(product_id) in self.data.get('wardrobe', [])

    def toggle_favorite(self, product_id: Any) -> bool:
        pid = _as_id(product_id)
        wardrobe = self.data.setdefault('wardrobe', [])
        if pid in wardrobe:
            wardrobe.remove(pid)
            active = False
        else:
            wardrobe.insert(0, pid)
            self.data['wardrobe'] = _dedupe(wardrobe, MAX_WARDROBE_ITEMS)
            active = True
        self.save()
        return active

    def clear_wardrobe(self) -> None:
        self.data['wardrobe'] = []
        self.save()

    # ----------------------------------------------------------------- history
    @property
    def recently_viewed(self) -> List[str]:
        return list(self.data.get('recently_viewed', []))

    def record_view(self, product_id: Any) -> None:
        pid = _as_id(product_id)
        if not pid:
            return
        viewed = [p for p in self.data.get('recently_viewed', []) if p != pid]
        viewed.insert(0, pid)
        self.data['recently_viewed'] = viewed[:MAX_RECENTLY_VIEWED]
        self.save()

    @property
    def recent_searches(self) -> List[Dict[str, Any]]:
        return list(self.data.get('recent_searches', []))

    def record_search(self, query: str, mode: str = 'text') -> None:
        query = (query or '').strip()
        if not query:
            return
        entry = {'query': query, 'mode': mode, 'at': _now()}
        history = [
            item for item in self.data.get('recent_searches', [])
            if not (str(item.get('query', '')).strip().lower() == query.lower() and item.get('mode') == mode)
        ]
        history.insert(0, entry)
        self.data['recent_searches'] = history[:MAX_RECENT_SEARCHES]
        self.save()

    def clear_searches(self) -> None:
        self.data['recent_searches'] = []
        self.save()

    # ------------------------------------------------------------------ outfits
    @property
    def saved_outfits(self) -> List[Dict[str, Any]]:
        return list(self.data.get('saved_outfits', []))

    def save_outfit(self, name: str, product_ids: Iterable[Any]) -> Dict[str, Any]:
        outfit = {
            'id': secrets.token_hex(4),
            'name': (name or 'Set đồ').strip() or 'Set đồ',
            'items': _dedupe(product_ids, 12),
            'created_at': _now(),
        }
        outfits = self.data.setdefault('saved_outfits', [])
        outfits.insert(0, outfit)
        self.data['saved_outfits'] = outfits[:MAX_SAVED_OUTFITS]
        self.save()
        return outfit

    def remove_outfit(self, outfit_id: str) -> None:
        self.data['saved_outfits'] = [
            item for item in self.data.get('saved_outfits', [])
            if str(item.get('id')) != str(outfit_id)
        ]
        self.save()

    # ------------------------------------------------------------ portability
    def export_json(self) -> str:
        payload = {
            'version': STATE_VERSION,
            'wardrobe': self.wardrobe,
            'saved_outfits': self.saved_outfits,
            'exported_at': _now(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def import_json(self, raw: str) -> int:
        try:
            payload = json.loads(raw)
        except Exception:
            raise ValueError('Tệp sao lưu không hợp lệ (JSON lỗi).')
        if not isinstance(payload, dict):
            raise ValueError('Tệp sao lưu không hợp lệ.')
        added = 0
        incoming = _dedupe(payload.get('wardrobe', []) or [], MAX_WARDROBE_ITEMS)
        current = self.data.setdefault('wardrobe', [])
        for pid in incoming:
            if pid not in current:
                current.append(pid)
                added += 1
        outfits = self.data.setdefault('saved_outfits', [])
        known = {str(o.get('id')) for o in outfits}
        for outfit in payload.get('saved_outfits', []) or []:
            if isinstance(outfit, dict) and str(outfit.get('id')) not in known:
                outfits.insert(0, outfit)
        self.save()
        return added
