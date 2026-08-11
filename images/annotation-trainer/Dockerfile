FROM python:3.12-slim

RUN pip install --no-cache-dir \
    torch==2.7.1 \
    mlflow==3.15.1

WORKDIR /app
COPY src/mlforge_platform /app/mlforge_platform
ENV PYTHONPATH=/app
