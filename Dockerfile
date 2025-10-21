# ── Builder ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

# System deps you may need (sqlite, gcc for faiss wheels if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Minimal OS deps (sqlite for your DB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy app source
COPY app ./app
COPY frontend ./frontend
COPY .env.example ./.env.example
# If you have a schema or seed scripts, copy them too:
# COPY app/db/schema.sql app/db/

# Expose default port
EXPOSE 8000

# Healthcheck (FastAPI /healthz)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# Start
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
