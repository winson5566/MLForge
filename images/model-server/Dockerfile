FROM mlforge/mlflow:3.15.1

RUN pip install --no-cache-dir fastapi==0.128.0 "uvicorn[standard]==0.40.0" prometheus-client==0.24.1

COPY app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8080"]
