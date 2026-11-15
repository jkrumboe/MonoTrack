# Analysis server (python/server.py) with the analyzer on an NVIDIA GPU.
#
#   docker compose up -d --build        # see compose.yaml; http://<host>:8765
#
# Needs the NVIDIA driver and NVIDIA Container Toolkit on the host (Docker
# Desktop on Windows: WSL 2 backend, GPU support is built in).

# ── Web app (dist/), so the container is the whole service ───────────────────
FROM node:22-bookworm-slim AS web
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.ts tsconfig*.json ./
COPY public ./public
COPY src ./src
# vite build without `tsc`: a type error in work in progress should not stop
# the analysis server from building. `npm run typecheck` is the place for that.
RUN npx vite build

# ── Analyzer + server ────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    VB_DATA=/data/server \
    TENNIS3D_COURTS=/data/courts \
    HF_HOME=/cache/huggingface \
    YOLO_CONFIG_DIR=/cache/ultralytics

# ffmpeg: phone recordings are re-encoded to a constant frame rate.
# libgl1/libglib2.0-0/libegl1/libgles2: OpenCV (pulled in by ultralytics) and mediapipe.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 curl \
 && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 wheels, the same build as the dev venv. The CUDA runtime comes in
# the wheels; the host only needs the driver.
# The pip cache mount keeps finished wheels across attempts, so a dropped
# connection retries only what is missing (and stays out of the image).
RUN --mount=type=cache,target=/root/.cache/pip \
    for i in 1 2 3; do pip install torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu128 && break; [ $i = 3 ] && exit 1; done

WORKDIR /app
COPY python/requirements.txt python/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    for i in 1 2 3; do pip install -r python/requirements.txt && break; [ $i = 3 ] && exit 1; done

# Weights, in their own layer so code changes don't download them again.
# Each fetch may fail (Google Drive rate limits gdown); the analyzer then
# tries again on first use, into the weights volume.
# The detectors save weights next to their module (python/models/), so they
# run from /app/python; the full COPY below replaces these files with the same ones.
COPY python/ball_tracknet.py python/ball_tracknetv3.py python/ball_vballnet.py /app/python/
RUN cd /app && python -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); YOLO('yolov8m-pose.pt')" \
 && cd /app/python && python -c "\
import ball_tracknet, ball_tracknetv3, ball_vballnet; \
ball_tracknet.ensure_weights(); \
ball_tracknetv3.ensure_weights(); \
[ball_vballnet.ensure_weights(n) for n in ball_vballnet.MODELS]" \
 && rm -rf /app/python/__pycache__ \
 && ls -la /app/*.pt /app/python/models || true

COPY python ./python
COPY --from=web /src/dist ./dist

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS -H "Authorization: Bearer ${VB_TOKEN}" http://127.0.0.1:8765/api/health || exit 1
CMD ["python", "python/server.py", "--host", "0.0.0.0", "--port", "8765"]
