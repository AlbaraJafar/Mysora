import os

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms, models

device = torch.device("cpu")

# CPU-friendly defaults (override with OMP_NUM_THREADS / MKL_NUM_THREADS)
try:
    _threads = max(1, int(os.environ.get("MYSORA_TORCH_THREADS", "4")))
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(max(1, _threads // 2))
except Exception:
    pass

num_classes = 31

MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/Albaraajaafar/mysora-model/resolve/main/MysoraBestModel.pth"
)

# Hugging Face CDN sometimes throttles anonymous clients with default User-Agent; avoid tiny/HTML bodies.
_HF_HEADERS = {
    "User-Agent": "MysoraAPI/1.0 (+https://huggingface.co/Albaraajaafar/mysora-model)",
    "Accept": "*/*",
}

# Real checkpoint ~90MB; HTML error pages are usually small
_MIN_CHECKPOINT_BYTES = 8_000_000

# build the same architecture used during training
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# load checkpoint — deployment: set MYSORA_MODEL_PATH to absolute path of .pth file
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]


def _canonical_default_model_path() -> Path:
    return (_PROJECT_ROOT / "model" / "MysoraBestModel.pth").resolve()


def _resolve_checkpoint_path() -> Path:
    env_path = os.environ.get("MYSORA_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        else:
            p = p.resolve()
        return p
    return _canonical_default_model_path()


def _validate_checkpoint_bytes(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Model file missing: {path}")
    sz = path.stat().st_size
    if sz < _MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Model file too small ({sz} bytes); expected a full .pth (>{_MIN_CHECKPOINT_BYTES}). "
            "Check Hugging Face URL and network."
        )
    with open(path, "rb") as f:
        head = f.read(512)
    low = head.lower()
    if b"<html" in low or b"<!doctype html" in low:
        raise RuntimeError(
            f"Model path contains HTML, not a PyTorch checkpoint: {path}. "
            "Fix the download URL or HF repo visibility."
        )
    if head.startswith(b"PK"):
        return
    if len(head) >= 1 and head[0] == 0x80:
        return
    raise RuntimeError(f"File does not look like a PyTorch .pth (bad header): {path}")


def _download_model(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading model...", flush=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(
            MODEL_DOWNLOAD_URL,
            stream=True,
            timeout=(30, 600),
            headers=_HF_HEADERS,
        ) as resp:
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise RuntimeError(
                    f"Model download failed (HTTP {resp.status_code}): {MODEL_DOWNLOAD_URL}"
                ) from e
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(dest)
        _validate_checkpoint_bytes(dest)
    except requests.RequestException as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download model from {MODEL_DOWNLOAD_URL}: {e}"
        ) from e
    except OSError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to save model to '{dest}': {e}"
        ) from e
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model download completed but file is missing or empty: {dest}"
        )
    print("Model downloaded", flush=True)


checkpoint_path_obj = _resolve_checkpoint_path()
if not checkpoint_path_obj.is_file():
    if checkpoint_path_obj.resolve() == _canonical_default_model_path():
        _download_model(checkpoint_path_obj)
    else:
        raise RuntimeError(
            f"MYSORA_MODEL_PATH points to a missing file: {checkpoint_path_obj}"
        )

checkpoint_path = checkpoint_path_obj.as_posix()

_validate_checkpoint_bytes(checkpoint_path_obj)

try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        raise RuntimeError(
            "Checkpoint has no 'model_state_dict' or 'state_dict' key; "
            f"keys: {list(checkpoint.keys())[:25] if isinstance(checkpoint, dict) else type(checkpoint)}"
        )
    model.load_state_dict(state)
except Exception as e:
    raise RuntimeError(
        f"Failed to load model checkpoint at '{checkpoint_path}'. "
        "Make sure the file exists and matches the trained architecture."
    ) from e

model.to(device)
model.eval()

classes = [
    "Ain", "Al", "Alef", "Beh", "Dad", "Dal", "Feh", "Ghain", "Hah", "Heh",
    "Jeem", "Kaf", "Khah", "Laa", "Lam", "Meem", "Noon", "Qaf", "Reh", "Sad",
    "Seen", "Sheen", "Tah", "Teh", "Teh_Marbuta", "Thal", "Theh", "Waw", "Yeh", "Zah", "Zain",
]

# ImageNet normalize (channel-first, float32) — avoids PIL for 224×224 grayscale path
_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

# Fallback path for non-standard sizes (uses PIL like original training)
_transform_legacy = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict_proba_gray224(gray_u8: np.ndarray):
    """
    Fast path: grayscale uint8 (224, 224), contiguous. No BGR/RGB/PIL.
    Returns (classes, probs numpy).
    """
    if gray_u8.dtype != np.uint8 or gray_u8.shape != (224, 224):
        raise ValueError("predict_proba_gray224 expects uint8 (224,224)")
    if not gray_u8.flags["C_CONTIGUOUS"]:
        gray_u8 = np.ascontiguousarray(gray_u8)

    # [1,1,H,W] float in [0,1] -> expand to 3ch -> normalize -> forward
    x = torch.from_numpy(gray_u8).to(device=device, dtype=torch.float32).div_(255.0)
    x = x.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    x = x.expand(1, 3, gray_u8.shape[0], gray_u8.shape[1]).contiguous()
    x = (x - _mean) / _std

    with torch.inference_mode():
        output = model(x)

    probabilities = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    return classes, probabilities


def predict_proba(frame):
    """
    Returns (labels, probabilities).
    Uses fast tensor path for 224×224 grayscale; legacy PIL path otherwise.
    """
    if frame.ndim == 2 and frame.shape == (224, 224) and frame.dtype == np.uint8:
        return predict_proba_gray224(frame)
    if frame.ndim == 2:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = _transform_legacy(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(t)
    probabilities = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    return classes, probabilities


def predict_letter(frame):
    labels, probs = predict_proba(frame)
    pred = int(probs.argmax())
    confidence = float(probs[pred])
    return labels[pred], confidence
