# Mysora deployment (Railway + Vercel)

## Backend (Railway)

1. Create a Railway service from this repo.
2. Set **Root Directory** to `Quran 2/Quran` (or the folder that contains `Dockerfile` and `api_main.py`).
3. Add environment variables:

| Variable | Description |
|----------|-------------|
| `MYSORA_MODEL_PATH` | Absolute path to `MysoraBestModel.pth` inside the container (optional if you ship `model/MysoraBestModel.pth` in the image). |
| `MYSORA_CORS_ORIGINS` | Comma-separated list of allowed origins, e.g. `https://your-app.vercel.app`. Use `*` for any origin (default). |
| `HAND_LANDMARKER_MODEL` | Optional path to `hand_landmarker.task` if not using `scripts/hand_landmarker.task`. |
| `MYSORA_TORCH_THREADS` | Optional CPU thread count (default `4`). |

4. **Model file:** Either commit `model/MysoraBestModel.pth` (large) or mount storage and set `MYSORA_MODEL_PATH=/data/MysoraBestModel.pth`. Same for Mediapipe `hand_landmarker.task` under `scripts/` or `HAND_LANDMARKER_MODEL`.

5. The container listens on **`0.0.0.0`** and **`PORT`** (Railway injects `PORT`; local Docker defaults to **8000**).

6. Health check: **`GET /health`**

### Local Docker

```bash
docker build -t mysora-api .
docker run -p 8000:8000 -e MYSORA_CORS_ORIGINS=https://your-app.vercel.app mysora-api
```

## Frontend (Vercel)

1. Create a Vercel project; set **Root Directory** to `Quran 2/Quran/static` (this folder contains `index.html`).
2. Set **Framework Preset** to “Other” (static site).
3. Configure the API URL so the browser calls Railway, not Vercel:

   - Edit `index.html` and `fatiha.html`: set  
     `<meta name="mysora-api-base" content="https://YOUR-SERVICE.up.railway.app" />`  
     (no trailing slash), **or**
   - Inject before `app.js`:  
     `window.MYSORA_API_BASE = 'https://YOUR-SERVICE.up.railway.app';`

4. Redeploy after changing the meta tag.

5. Ensure Railway `MYSORA_CORS_ORIGINS` includes your Vercel URL(s).

## Production `requirements.txt`

- **`opencv-python-headless`**: no GUI deps (servers / Docker).
- **PyTorch** installed from the **CPU** extra index (see top of `requirements.txt`).

## Headless / no local paths

- **OpenCV:** `opencv-python-headless` only in `requirements.txt`.
- **`camera_test.py`:** font uses `MYSORA_FONT_PATH` or system fonts / default glyph (no Windows-only path required for the API).
