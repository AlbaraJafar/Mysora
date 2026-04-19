# Mysora API — Railway / any container host
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MYSORA_TORCH_THREADS=4 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libraries for opencv-python-headless (wheels link libxcb, libX11, etc.; not GUI toolkits)
# and Mediapipe. No GTK/Qt; keep --no-install-recommends and drop apt lists for smaller image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libxcb-shm0 \
    libx11-6 \
    libx11-xcb1 \
    libxau6 \
    libxdmcp6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libbsd0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Model: bind mount or bake in image; or set MYSORA_MODEL_PATH to a persistent path
EXPOSE 8000

# Railway sets PORT; api_main.py reads it (default 8000 for local Docker)
CMD ["python", "api_main.py"]
