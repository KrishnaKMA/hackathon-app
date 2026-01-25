FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for local, Azure will override
EXPOSE 8000

# IMPORTANT: use shell form so $PORT is expanded
CMD gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --timeout 180 \
    --bind 0.0.0.0:${PORT:-8000}
