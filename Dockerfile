FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps (faiss + sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HF Spaces expects app on $PORT
EXPOSE 7860

CMD ["bash", "start.sh"]
