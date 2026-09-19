FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY src /app/src
COPY config /app/config
COPY data/features /app/data/features
COPY models /app/models
COPY mlflow.db /app/mlflow.db
COPY mlruns /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]