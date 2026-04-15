# Mysora API — Railway / any container host
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MYSORA_TORCH_THREADS=4 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal deps for OpenCV headless + Mediapipe wheels on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Model: bind mount or bake in image; or set MYSORA_MODEL_PATH to a persistent path
EXPOSE 8000

# Railway provides PORT; default 8000 for local Docker (shell expands ${PORT:-8000})
CMD ["sh", "-c", "exec uvicorn api_main:app --host 0.0.0.0 --port ${PORT:-8000}"]
