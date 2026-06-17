"""Community sign-language data collection (sessions + clips on disk)."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from mysora_letters import letter_map

_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_BASE_DIR / "outputs")))
OUTPUTS_DIR = DATA_DIR
SESSIONS_DIR = DATA_DIR / "sessions"
CLIPS_DIR = DATA_DIR / "clips"
PROGRESS_FILE = DATA_DIR / "progress.json"

TARGET_PER_LETTER = 100

COLLECT_PRIORITY: Dict[str, List[str]] = {
    "critical": ["ح", "و", "ق", "ب"],
    "high": ["ث", "ز", "ه", "ل", "ط", "ظ"],
    "medium": ["ا", "ت", "ن"],
}

_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "normal": 3}

_lock = threading.Lock()


def _init_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)


_init_dirs()


def _ensure_dirs() -> None:
    _init_dirs()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def letter_priority(letter: str) -> str:
    for tier, letters in COLLECT_PRIORITY.items():
        if letter in letters:
            return tier
    return "normal"


def _all_arabic_letters() -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for ar in letter_map.values():
        for ch in ar:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
    return out


def _priority_letters_flat() -> List[str]:
    flat: List[str] = []
    for letters in COLLECT_PRIORITY.values():
        flat.extend(letters)
    return flat


def _count_clips_by_letter() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not CLIPS_DIR.is_dir():
        return counts
    for label_dir in CLIPS_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        n = sum(1 for p in label_dir.glob("*.json") if p.is_file())
        if n:
            counts[label_dir.name] = n
    return counts


def _sessions_today() -> int:
    if not SESSIONS_DIR.is_dir():
        return 0
    today = _today_utc()
    n = 0
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created = str(data.get("created_at", ""))[:10]
            if created == today:
                n += 1
        except (json.JSONDecodeError, OSError):
            continue
    return n


def create_session(
    *,
    signer_type: Literal["deaf", "hearing"],
    dominant_hand: Literal["right", "left"],
    experience_years: int,
) -> Dict[str, str]:
    _ensure_dirs()
    session_id = secrets.token_hex(16)
    signer_id = hashlib.sha256(
        f"{session_id}:{signer_type}:{dominant_hand}:{experience_years}:{secrets.token_hex(8)}".encode()
    ).hexdigest()[:24]

    payload = {
        "session_id": session_id,
        "signer_id": signer_id,
        "signer_type": signer_type,
        "dominant_hand": dominant_hand,
        "experience_years": int(experience_years),
        "created_at": _utc_now_iso(),
        "clips_count": 0,
    }
    path = SESSIONS_DIR / f"{session_id}.json"
    with _lock:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"session_id": session_id, "signer_id": signer_id}


def save_clip(
    *,
    frame_data: str,
    label: str,
    label_type: Literal["letter", "word", "sentence"],
    session_id: str,
    signer_id: str,
    hand_orientation: Literal["front", "back"],
    confidence: float,
    thumbnail_b64: str | None = None,
) -> Dict[str, str]:
    _ensure_dirs()
    if not label.strip():
        raise ValueError("label is required")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    clip_id = f"{session_id}_{ts}"
    label_dir = CLIPS_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_dir / f"{clip_id}.json"

    parsed_frame: object = frame_data
    stripped = frame_data.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed_frame = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_frame = frame_data

    clip = {
        "clip_id": clip_id,
        "session_id": session_id,
        "signer_id": signer_id,
        "label": label,
        "label_type": label_type,
        "hand_orientation": hand_orientation,
        "confidence": float(confidence),
        "frame_data": parsed_frame,
        "created_at": _utc_now_iso(),
    }

    if thumbnail_b64:
        import base64

        thumb_dir = CLIPS_DIR / label / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{clip_id}.png"
        thumb_path.write_bytes(base64.b64decode(thumbnail_b64))
        clip["thumbnail_path"] = str(thumb_path)

    with _lock:
        out_path.write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
        sess_path = SESSIONS_DIR / f"{session_id}.json"
        if sess_path.is_file():
            try:
                sess = json.loads(sess_path.read_text(encoding="utf-8"))
                sess["clips_count"] = int(sess.get("clips_count", 0)) + 1
                sess["last_clip_at"] = _utc_now_iso()
                sess_path.write_text(
                    json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except (json.JSONDecodeError, OSError):
                pass

    return {"clip_id": clip_id, "status": "saved"}


def get_progress() -> Dict[str, object]:
    by_letter = _count_clips_by_letter()
    total = sum(by_letter.values())
    return {
        "total_clips": total,
        "by_letter": by_letter,
        "target_per_letter": TARGET_PER_LETTER,
        "sessions_today": _sessions_today(),
    }


def _next_letter_candidate(
    counts: Dict[str, int],
) -> Tuple[str, int, str]:
    """Pick letter with fewest clips; priority tier breaks ties."""
    priority_set = _priority_letters_flat()
    pool = priority_set if any(counts.get(l, 0) < TARGET_PER_LETTER for l in priority_set) else _all_arabic_letters()

    def sort_key(letter: str) -> Tuple[int, int, str]:
        tier = _PRIORITY_ORDER.get(letter_priority(letter), 3)
        return (tier, counts.get(letter, 0), letter)

    letter = min(pool, key=sort_key)
    count = counts.get(letter, 0)
    return letter, count, letter_priority(letter)


def get_next_letter() -> Dict[str, object]:
    counts = _count_clips_by_letter()
    letter, count, priority = _next_letter_candidate(counts)
    return {
        "letter": letter,
        "current_count": count,
        "target": TARGET_PER_LETTER,
        "priority": priority,
    }
