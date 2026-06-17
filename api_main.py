from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Optional, List
import threading
import time
import copy

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from scripts.inference import predict_proba
from scripts.hand_crop import prepare_for_inference
from gesture_engine import GestureStabilizer
from mysora_letters import letter_map, target_fatiha, fatiha_verses
from camera_manager import CameraManager
from collect_data import (
    create_session,
    get_next_letter,
    get_progress,
    save_clip,
)

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"

# CORS: set MYSORA_CORS_ORIGINS=https://app.vercel.app,https://mysite.com (comma-separated).
# Default * allows any origin (fine for many APIs); tighten in production if you prefer.
def _cors_origins() -> List[str]:
    raw = os.environ.get("MYSORA_CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Mysora Gesture API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_sessions_lock = threading.Lock()
_sessions: Dict[str, Dict[str, object]] = {}

_camera = CameraManager(camera_index=1)  # matches your original script default
_leaderboard_lock = threading.Lock()
_leaderboard: List[Dict[str, object]] = []  # in-memory leaderboard

# (Removed frame-similarity skip: holding a steady sign kept MAD low and blocked inference,
# so the stabilizer never received enough updates to emit.)


def _get_session(client_id: str) -> Dict[str, object]:
    with _sessions_lock:
        sess = _sessions.get(client_id)
        if sess is None:
            sess = {
                "stabilizer": GestureStabilizer(num_classes=31),
                "word": "",  # accepted text only
                "pos": 0,     # position inside target
                "target": target_fatiha,
                "_last_bundle": None,
                "_last_gray": None,
                "_last_infer_mono": 0.0,
                "_infer_tick": 0,
            }
            _sessions[client_id] = sess
        else:
            sess.setdefault("_last_bundle", None)
            sess.setdefault("_last_gray", None)
            sess.setdefault("_last_infer_mono", 0.0)
            sess.setdefault("_infer_tick", 0)
        return sess


def _want_inference(sess: Dict[str, object], now: float) -> bool:
    """~150–250 ms effective rate; every 2nd/3rd request can run a bit sooner."""
    last = float(sess.get("_last_infer_mono", 0.0))
    tick = int(sess.get("_infer_tick", 0)) + 1
    sess["_infer_tick"] = tick
    if last <= 0.0:
        return True
    elapsed_ms = (now - last) * 1000.0
    if elapsed_ms >= 240.0:
        return True
    if elapsed_ms >= 160.0 and tick % 2 == 0:
        return True
    if elapsed_ms >= 110.0 and tick % 3 == 0:
        return True
    return False


def _build_predict_bundle(
    sess: Dict[str, object],
    pred,
    emitted_arabic: Optional[str],
    attempt_info: Optional[Dict[str, object]],
    *,
    hand_present: bool,
    inference_skipped: bool,
) -> Dict[str, object]:
    word = str(sess["word"])
    target = str(sess["target"])
    pos = _advance_over_spaces(target, int(sess.get("pos", 0)))
    complete = pos >= len(target)
    return {
        "raw": {
            "label": pred.raw_label,
            "confidence": pred.raw_confidence,
            "arabic_preview": letter_map.get(pred.raw_label, ""),
        },
        "stable": {
            "emitted_label": pred.stable_label,
            "emitted_arabic": emitted_arabic,
            "ema_confidence": pred.stable_confidence,
            "margin": pred.margin,
        },
        "word": word,
        "progress_index": pos,
        "attempt": attempt_info,
        "progress_correct": True,
        "complete": complete,
        "hand_present": hand_present,
        "inference_skipped": inference_skipped,
    }


def _run_full_inference(sess: Dict[str, object], gray224: np.ndarray, hand_present: bool, now: float):
    labels, probs = predict_proba(gray224)
    stabilizer: GestureStabilizer = sess["stabilizer"]  # type: ignore[assignment]
    pred = stabilizer.update(labels=list(labels), probs=probs)

    emitted_arabic: Optional[str] = None
    attempt_info: Optional[Dict[str, object]] = None
    if pred.stable_label is not None:
        emitted_arabic = letter_map.get(pred.stable_label, "")
        if emitted_arabic:
            attempt_info = _apply_attempt(sess, emitted_arabic)

    sess["_last_gray"] = gray224.copy()
    sess["_last_infer_mono"] = now
    bundle = _build_predict_bundle(
        sess, pred, emitted_arabic, attempt_info, hand_present=hand_present, inference_skipped=False
    )
    sess["_last_bundle"] = copy.deepcopy(bundle)
    return bundle


def _reuse_last_bundle(sess: Dict[str, object], *, hand_present: bool, skipped: bool) -> Dict[str, object]:
    cached = sess.get("_last_bundle")
    if not isinstance(cached, dict):
        return {}
    out = copy.deepcopy(cached)
    out["hand_present"] = hand_present
    out["inference_skipped"] = skipped
    # Do not re-emit the same stable letter across skipped frames (avoids duplicate UI / stats)
    st = out.get("stable")
    if isinstance(st, dict):
        st = dict(st)
        st["emitted_label"] = None
        st["emitted_arabic"] = None
        out["stable"] = st
    out["attempt"] = None
    return out


def _advance_over_spaces(target: str, pos: int) -> int:
    while pos < len(target) and target[pos] == " ":
        pos += 1
    return pos


def _apply_attempt(sess: Dict[str, object], attempted: str) -> Dict[str, object]:
    """
    Apply attempted Arabic text to the session (one or more codepoints, e.g. "ال", "لا").
    Accepts only if it exactly matches the next substring of the target (after spaces).
    """
    target = str(sess["target"])
    pos = int(sess.get("pos", 0))
    word = str(sess.get("word", ""))

    pos = _advance_over_spaces(target, pos)
    expected = target[pos] if pos < len(target) else ""

    if pos >= len(target):
        return {"accepted": False, "expected": "", "attempted": attempted, "complete": True}

    if not attempted:
        return {"accepted": False, "expected": expected, "attempted": attempted, "complete": False}

    # Multi-codepoint signs (letter_map: Al -> "ال", Laa -> "لا")
    if len(attempted) > 1:
        end = pos + len(attempted)
        if end <= len(target) and target[pos:end] == attempted:
            word += attempted
            pos = end
            sess["word"] = word
            sess["pos"] = pos
            pos2 = _advance_over_spaces(target, pos)
            expected2 = target[pos2] if pos2 < len(target) else ""
            complete = pos2 >= len(target)
            return {"accepted": True, "expected": expected2, "attempted": attempted, "complete": complete}
        return {"accepted": False, "expected": expected, "attempted": attempted, "complete": False}

    if attempted == expected:
        word += attempted
        pos += 1
        sess["word"] = word
        sess["pos"] = pos
        pos2 = _advance_over_spaces(target, pos)
        expected2 = target[pos2] if pos2 < len(target) else ""
        complete = (pos2 >= len(target))
        return {"accepted": True, "expected": expected2, "attempted": attempted, "complete": complete}

    # wrong attempt: do not mutate accepted word/pos
    return {"accepted": False, "expected": expected, "attempted": attempted, "complete": False}


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


@app.get("/health")
def health():
    """Liveness/readiness for Railway and other hosts."""
    from pathlib import Path

    import scripts.inference as inf

    p = Path(inf.checkpoint_path)
    nbytes = p.stat().st_size if p.is_file() else 0
    return {
        "ok": True,
        "service": "mysora-api",
        "inference": "cpu",
        "model_checkpoint_bytes": nbytes,
    }


@app.get("/config/fatiha")
def get_fatiha_config():
    return {
        "surah": "Al-Fatiha",
        "target": target_fatiha,
        "verses": fatiha_verses,
    }


@app.post("/predict")
async def predict(
    client_id: str = Query(...),
    image: UploadFile = File(...),
):
    data = await image.read()
    frame = _decode_image_bytes(data)
    crop, hand_ok = prepare_for_inference(frame)
    sess = _get_session(client_id)
    now = time.monotonic()

    if sess.get("_last_bundle") is not None and not _want_inference(sess, now):
        return _reuse_last_bundle(sess, hand_present=hand_ok, skipped=True)

    return _run_full_inference(sess, crop, hand_ok, now)


@app.post("/reset")
def reset(client_id: str = Query(...)):
    sess = _get_session(client_id)
    stabilizer: GestureStabilizer = sess["stabilizer"]  # type: ignore[assignment]
    stabilizer.reset()
    sess["word"] = ""
    sess["pos"] = 0
    sess["_last_bundle"] = None
    sess["_last_gray"] = None
    sess["_last_infer_mono"] = 0.0
    sess["_infer_tick"] = 0
    return {"ok": True}


@app.get("/camera/predict")
def camera_predict(client_id: str = Query(...), camera_index: int = 1):
    # allow switching camera without restarting
    _camera.set_camera(camera_index)
    frame = _camera.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")
    crop, hand_ok = prepare_for_inference(frame)
    sess = _get_session(client_id)
    now = time.monotonic()

    if sess.get("_last_bundle") is not None and not _want_inference(sess, now):
        return _reuse_last_bundle(sess, hand_present=hand_ok, skipped=True)

    return _run_full_inference(sess, crop, hand_ok, now)


@app.post("/leaderboard/submit")
async def leaderboard_submit(
    name: str,
    duration_seconds: float,
    accuracy: float,
):
    entry = {
        "name": name.strip() or "Anonymous",
        "duration_seconds": float(duration_seconds),
        "accuracy": float(accuracy),
    }
    with _leaderboard_lock:
        _leaderboard.append(entry)
        _leaderboard.sort(key=lambda e: (-e["accuracy"], e["duration_seconds"]))
        del _leaderboard[20:]
    return {"ok": True}


@app.get("/leaderboard/top")
def leaderboard_top(limit: int = 10):
    with _leaderboard_lock:
        return {"entries": _leaderboard[: max(1, min(limit, 20))]}


# ============ Data collection (community dataset) ============


class CollectSessionBody(BaseModel):
    signer_type: str = Field(..., pattern="^(deaf|hearing)$")
    dominant_hand: str = Field(..., pattern="^(right|left)$")
    experience_years: int = Field(..., ge=0, le=80)


@app.post("/collect/session")
def collect_session(body: CollectSessionBody):
    try:
        return create_session(
            signer_type=body.signer_type,  # type: ignore[arg-type]
            dominant_hand=body.dominant_hand,  # type: ignore[arg-type]
            experience_years=body.experience_years,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/collect/clip")
async def collect_clip(
    frame_data: str = Form(...),
    label: str = Form(...),
    label_type: str = Form("letter"),
    session_id: str = Form(...),
    signer_id: str = Form(...),
    hand_orientation: str = Form("front"),
    confidence: float = Form(0.0),
    thumbnail_b64: Optional[str] = Form(None),
):
    if label_type not in ("letter", "word", "sentence"):
        raise HTTPException(status_code=400, detail="Invalid label_type")
    if hand_orientation not in ("front", "back"):
        raise HTTPException(status_code=400, detail="Invalid hand_orientation")
    try:
        return save_clip(
            frame_data=frame_data,
            label=label.strip(),
            label_type=label_type,  # type: ignore[arg-type]
            session_id=session_id.strip(),
            signer_id=signer_id.strip(),
            hand_orientation=hand_orientation,  # type: ignore[arg-type]
            confidence=confidence,
            thumbnail_b64=thumbnail_b64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/collect/progress")
def collect_progress():
    return get_progress()


@app.get("/collect/next-letter")
def collect_next_letter():
    return get_next_letter()


@app.get("/collect/admin")
def collect_admin(password: str = ""):
    admin_pw = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_pw or password != admin_pw:
        raise HTTPException(status_code=403, detail="Forbidden")

    progress = get_progress()
    total = progress.get("total_clips", 0)
    by_letter = progress.get("by_letter", {})
    target = 100

    rows = ""
    priority_letters = ["ح", "و", "ق", "ب", "ث", "ز", "ه", "ل", "ط", "ظ", "أ", "ت", "ن"]
    for letter in priority_letters:
        count = by_letter.get(letter, 0)
        pct = min(int(count / target * 100), 100)
        color = "#2ea043" if pct >= 100 else "#d29922" if pct > 0 else "#f85149"
        rows += f"""
          <tr>
            <td style='font-size:24px;
                       text-align:center'>{letter}</td>
            <td>
              <div style='background:#30363d;
                          border-radius:4px;height:20px'>
                <div style='background:{color};
                            width:{pct}%;height:20px;
                            border-radius:4px'></div>
              </div>
            </td>
            <td style='text-align:center;
                       color:{color}'>{count}/{target}</td>
          </tr>"""

    html = f"""<!DOCTYPE html>
      <html dir='rtl' lang='ar'>
      <head>
        <meta charset='UTF-8'>
        <title>Mysora Admin</title>
        <style>
          body{{background:#0d1117;color:#f0f6fc;
               font-family:Calibri;padding:2rem}}
          table{{width:100%;border-collapse:collapse}}
          td{{padding:12px;border-bottom:
              1px solid #21262d}}
          h1{{color:#00b4d8}}
          .stat{{background:#161b22;padding:1rem;
                 border-radius:8px;margin:1rem 0;
                 display:inline-block;min-width:150px;
                 text-align:center}}
          .stat-num{{font-size:2rem;
                     color:#2ea043;font-weight:bold}}
        </style>
      </head>
      <body>
        <h1>📊 Mysora Dataset Admin</h1>
        <div class='stat'>
          <div class='stat-num'>{total}</div>
          <div>Total Clips</div>
        </div>
        <div class='stat'>
          <div class='stat-num'>
            {progress.get('sessions_today', 0)}
          </div>
          <div>Sessions Today</div>
        </div>
        <h2>Letter Progress</h2>
        <table>{rows}</table>
      </body>
      </html>"""

    return HTMLResponse(html)


@app.post("/contact")
async def contact_form(
    subject: str = Form(...),
    message: str = Form(...),
):
    try:
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        to_email  = os.environ.get("CONTACT_EMAIL", smtp_user)

        if not smtp_user or not smtp_pass:
            print(f"[CONTACT] {subject}: {message}")
            return {"status": "ok"}

        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = to_email
        msg["Subject"] = f"[ميسورة] {subject}"
        body = (
            f"موضوع: {subject}\n\n"
            f"الرسالة:\n{message}\n\n"
            f"---\nأُرسلت من موقع ميسورة"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)

        return {"status": "ok"}

    except Exception as e:
        print(f"[CONTACT ERROR] {e}")
        raise HTTPException(status_code=500, detail="فشل الإرسال")


if _STATIC_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="static",
    )


@app.get("/")
def serve_index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend not bundled; deploy static/ or use Vercel for UI.")
    return FileResponse(str(index_path))


# Root-level assets so relative href="styles.css" / src="app.js" work when UI is served from Railway.
@app.get("/styles.css")
def _styles():
    p = _STATIC_DIR / "styles.css"
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="text/css")


@app.get("/app.js")
def _app_js():
    p = _STATIC_DIR / "app.js"
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="application/javascript")


@app.get("/fatiha.html")
def _fatiha_html():
    p = _STATIC_DIR / "fatiha.html"
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


@app.get("/collect.html")
def _collect_html():
    p = _STATIC_DIR / "collect.html"
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


@app.get("/collect.js")
def _collect_js():
    p = _STATIC_DIR / "collect.js"
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="application/javascript")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn

    uvicorn.run("api_main:app", host="0.0.0.0", port=port)
