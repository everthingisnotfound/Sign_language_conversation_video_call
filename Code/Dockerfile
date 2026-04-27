FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn sign_language_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
