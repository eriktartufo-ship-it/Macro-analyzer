FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dipendenze di sistema: libpq per psycopg, curl per healthcheck, gcc per eventuali wheel build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq5 \
      curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend /app/backend

WORKDIR /app/backend

# Cache FRED persistente (montata come volume in compose)
RUN mkdir -p /app/backend/.cache/fred

RUN chmod +x /app/backend/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/app/backend/entrypoint.sh"]
