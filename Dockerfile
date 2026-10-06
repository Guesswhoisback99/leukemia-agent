FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

RUN useradd --create-home --home-dir /home/app app
WORKDIR /app
RUN chown app:app /app
USER app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY --chown=app:app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --group api

COPY --chown=app:app app ./app

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
