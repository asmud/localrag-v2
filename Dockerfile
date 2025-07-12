# ================================
# STAGE 1: Base dependencies layer
# ================================
FROM python:3.11-slim AS base

# Set environment variables for all stages
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    wget \
    ca-certificates \
    # Tesseract OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ind \
    libtesseract-dev \
    # PaddleOCR system dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    # Additional libraries for stability
    libc6-dev \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ================================
# STAGE 2: Python dependencies
# ================================
FROM base AS python-deps

# Create app directory
WORKDIR /app

# Copy only dependency files first (for better caching)
COPY pyproject.toml ./

# Install Python dependencies (cached layer)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# ================================
# STAGE 3: Model downloads
# ================================
FROM python-deps AS model-cache

# Create models directory with proper permissions
RUN mkdir -p /app/models/ocr && \
    chmod -R 755 /app/models

# Copy model download script and dependencies
COPY utils/download_ocr_models.py ./utils/
COPY core/ ./core/
COPY .env.example .env

# Pre-download OCR models to cache them in the image layer
# This creates a cached layer with models that can be reused
RUN echo "Checking for existing OCR models..." && \
    python3 ./utils/download_ocr_models.py --validate-models --languages en id && \
    echo "✅ OCR models already exist, skipping download" || \
    ( echo "📥 Pre-downloading EasyOCR models..." && \
      python3 ./utils/download_ocr_models.py --easyocr-only --languages en id && \
      echo "✅ OCR models cached successfully" ) || \
    echo "⚠️  OCR model download failed, will download at runtime"

# ================================
# STAGE 4: Production image
# ================================
FROM python-deps AS production

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/migrations \
    && chmod -R 755 /app/models \
    && chown -R appuser:appuser /app

# Copy cached models from model-cache stage (if available)
COPY --from=model-cache --chown=appuser:appuser /app/models/ /app/models/

# Copy application code
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser prompts/ ./prompts/

# Switch to non-root user
USER appuser

# Create volume mount points for runtime data
VOLUME ["/app/data", "/app/logs"]

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]