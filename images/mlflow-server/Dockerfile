FROM python:3.12-slim

ARG MLFLOW_VERSION=3.15.1

RUN pip install --no-cache-dir \
    "mlflow-skinny==${MLFLOW_VERSION}" \
    "psycopg2-binary==2.9.11" \
    "boto3==1.42.34" \
    "Flask==3.1.3" \
    "Flask-Cors==6.0.5" \
    "gunicorn==26.0.0" \
    "SQLAlchemy==2.0.51" \
    "alembic==1.19.1"

EXPOSE 5000

ENTRYPOINT ["mlflow"]
