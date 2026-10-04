# syntax=docker/dockerfile:1
# RAGScope backend (FastAPI). Linux container: also sidesteps Windows application-control
# policies that block compiled Python extensions (spaCy, scikit-learn, ...) on the host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    DATA_DIR=/data \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles CUDA (several GB larger).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies only (from pyproject, incl. dev extras for tests); the source is copied below or
# bind-mounted in development, so code changes never reinstall packages.
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /tmp/requirements.txt \
    && pip install --retries 10 -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data /cache/huggingface \
    && chown -R app:app /data /cache /app
COPY --chown=app:app app ./app
COPY --chown=app:app tests ./tests
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
