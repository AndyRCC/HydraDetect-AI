# Universal image — works on Hugging Face Spaces (Docker SDK, port 7860),
# Render (Docker), Fly.io, or any container host.
FROM python:3.11-slim

WORKDIR /app

# System libs some wheels expect (lightgbm needs libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects 7860; Render/Fly inject $PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
