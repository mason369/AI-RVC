# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
ARG PYTHON_IMAGE=python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS player
WORKDIR /app/ui/multitrack
COPY ui/multitrack/package.json ui/multitrack/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/multitrack/build.mjs ./
COPY ui/multitrack/src ./src
COPY i18n /app/i18n
RUN mkdir -p dist && npm run build

FROM ${PYTHON_IMAGE} AS upstream
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /source
COPY tools/upstream_runtime.py tools/model_assets.py ./tools/
RUN python -c "from pathlib import Path; from tools.upstream_runtime import ensure_source; [ensure_source(Path('/source'), capability) for capability in ('vc', 'uvr5')]"

FROM ${PYTHON_IMAGE} AS dependencies
ARG VARIANT=cuda
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsamplerate0-dev libsndfile1 pkg-config git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY pre-requirements.txt requirements*.txt /build/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /build/pre-requirements.txt && \
    case "$VARIANT" in cpu) wheel=cpu ;; cuda) wheel=cu128 ;; *) echo "不支持的镜像类型: $VARIANT" >&2; exit 1 ;; esac && \
    printf 'torch==2.11.0+%s\ntorchvision==0.26.0+%s\ntorchaudio==2.11.0+%s\n' "$wheel" "$wheel" "$wheel" > /build/torch-constraints.txt && \
    pip install -c /build/torch-constraints.txt -r "/build/requirements_${VARIANT}.txt" \
      --extra-index-url "https://download.pytorch.org/whl/$wheel" && \
    pip check && pip freeze > /build/python-packages.txt

FROM ${PYTHON_IMAGE} AS runtime
ARG VARIANT=cuda
ARG VERSION=1.5.2
ARG REVISION=local
LABEL org.opencontainers.image.title="AI-RVC" \
      org.opencontainers.image.description="RVC voice conversion and Leap / MVSep song covers" \
      org.opencontainers.image.source="https://github.com/mason369/AI-RVC" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$REVISION
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg git libsamplerate0 libsndfile1 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 rvc && useradd --uid 1000 --gid 1000 --create-home rvc
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /build/python-packages.txt /opt/ai-rvc/python-packages.txt
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AI_RVC_VARIANT=$VARIANT AI_RVC_DEVICE=$VARIANT \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/cache/torch GRADIO_TEMP_DIR=/data/temp/gradio \
    GRADIO_ANALYTICS_ENABLED=False NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY run.py app.py LICENSE README.md VERSION ./
COPY configs ./configs
COPY i18n ./i18n
COPY infer ./infer
COPY lib ./lib
COPY models ./models
COPY rvc_mcp ./rvc_mcp
COPY tools ./tools
COPY ui ./ui
COPY docker ./docker
COPY --from=player /app/ui/multitrack/dist/player.html ./ui/multitrack/dist/player.html
COPY --from=upstream --chown=1000:1000 /source/_official_rvc_runtime ./_official_rvc_runtime
# The two pinned Git trees are cached independently of app edits and GPU dependencies.
RUN mkdir -p /opt/ai-rvc /data/assets /data/outputs /data/temp /data/logs /data/cache \
    && mv configs/config.json /opt/ai-rvc/default-config.json \
    && ln -s /data/config.json configs/config.json \
    && ln -s /data/assets assets && ln -s /data/outputs outputs \
    && ln -s /data/temp temp && ln -s /data/logs logs \
    && chown -R 1000:1000 /data
USER 1000:1000
VOLUME ["/data"]
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/docker/entrypoint.py"]
CMD ["serve"]
