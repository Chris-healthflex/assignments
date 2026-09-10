FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install libsndfile for soundfile WAV decoding (eliminating any ffmpeg dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache Whisper base model inside the container image
RUN python -c "import whisper; whisper.load_model('base')"

# Copy application source code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY README.md .
COPY IMPLEMENTATION.md .
COPY clinical_assessment.wav .

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
