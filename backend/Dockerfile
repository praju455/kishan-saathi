FROM ghcr.io/astral-sh/uv:0.11.32-python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN groupadd --system --gid 10001 krishisathi \
    && useradd --system --uid 10001 --gid krishisathi --home /app krishisathi \
    && chmod 0555 /app/docker-entrypoint.sh \
    && chown -R krishisathi:krishisathi /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["uv", "run", "--no-sync", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

CMD ["/app/docker-entrypoint.sh"]
