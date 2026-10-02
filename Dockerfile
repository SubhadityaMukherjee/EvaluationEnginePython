# syntax=docker/dockerfile:1
#
# Usage (the -f/--function argument is required):
#   docker build -t evaluation-engine .
#   docker run --rm evaluation-engine -f evaluate_run -id 1 -test
#   docker run --rm evaluation-engine            # prints usage (--help default)
#
FROM python:3.14-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential pkg-config \
  && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . /app

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
