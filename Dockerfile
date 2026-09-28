FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the image a lot smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

USER app
ENV HOME=/home/app DATA_DIR=/home/app/data PYTHONUNBUFFERED=1
# docs, models and the index are baked into the image so startup is fast
RUN docqa ingest

EXPOSE 8000
CMD ["sh", "-c", "docqa serve --port ${PORT:-8000}"]
