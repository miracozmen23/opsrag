# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX_URL=https://pypi.org/simple

ENV HOME=/home/opsrag \
    MODEL_CACHE_DIR=/app/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PROCESSED_CHUNKS_PATH=/app/data/processed/chunks.jsonl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 opsrag \
    && useradd --uid 10001 --gid opsrag --create-home --shell /usr/sbin/nologin opsrag

COPY pyproject.toml README.md ./
RUN mkdir -p app frontend \
    && touch app/__init__.py frontend/__init__.py \
    && python -m pip install \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        "torch>=2.2,<3.0" \
    && python -m pip install . \
    && rm -rf app frontend

COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY evaluation ./evaluation
COPY .streamlit ./.streamlit

RUN python -m pip install --no-deps . \
    && mkdir -p /app/.cache/huggingface /app/data/processed \
    && chown -R opsrag:opsrag /app /home/opsrag

USER opsrag

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
