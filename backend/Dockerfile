# ---------- Stage 1: builder ----------
# Compila wheel pesanti (pyarrow, statsmodels) con build-essential, poi
# scarta i tool di build dallo stage finale → image runtime piu' leggera.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps in --user (poi copiate nello stage runtime)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

# ---------- Stage 2: runtime ----------
# Slim runtime con solo libpq5 (driver postgres) + curl (healthcheck).
# Niente build-essential, niente cache pip → image molto piu' piccola.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libpq5 \
      curl \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Copia le deps Python compilate dal builder
COPY --from=builder /root/.local /root/.local

WORKDIR /app

# Copia SOLO cio' che serve in produzione (NO tests, scripts, .cache):
# - app/ (codice runtime)
# - migrations/ + alembic.ini (schema upgrade su startup)
# - seed/ (idempotent seeds)
# - entrypoint.sh
# .dockerignore esclude tutto il resto.
COPY backend/app /app/app
COPY backend/migrations /app/migrations
COPY backend/alembic.ini /app/alembic.ini
COPY backend/seed /app/seed
COPY backend/entrypoint.sh /app/entrypoint.sh

# Cache FRED persistente (montata come volume in compose)
RUN mkdir -p /app/.cache/fred \
 && chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
