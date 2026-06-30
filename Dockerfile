# =============================================================================
# ILLUSTRATIVE / REFERENCE IMAGE — part of the On-Prem showcase.
# -----------------------------------------------------------------------------
# Builds the churn serving API container as designed for On-Prem deployment
# (ARCHITECTURE.md 4). In the real pipeline the CI `build` job fetches the
# Production model from the MLflow registry into ./artifacts before running
# `docker build` (see .github/workflows/ci.yml). It is not built automatically
# in this showcase repo — the internal image registry and model registry are
# not provisioned here.
# =============================================================================

# Lightweight, On-Prem friendly image for the churn serving API.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source and the trained artefacts.
# Artefacts are NOT in Git (model registry / object store, see ARCHITECTURE.md
# 4); CI fetches the Production model into ./artifacts before this build. The
# .gitkeep keeps the dir present so this COPY never fails on a fresh checkout;
# the guard below fails the build loudly rather than ship an image with no model.
COPY src/ ./src/
COPY artifacts/ ./artifacts/
RUN test -f artifacts/model.pkl \
    || (echo "ERROR: artifacts/model.pkl missing. Fetch the Production model from the registry before 'docker build' (see ARCHITECTURE.md §4)." && exit 1)

# src is importable as a package; unbuffered logs for container stdout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Container-level health check hits the API readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
