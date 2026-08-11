FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git gh \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "dvc[s3]==3.67.1" boto3

WORKDIR /app
COPY src/mlforge_platform /app/mlforge_platform
ENV PYTHONPATH=/app
