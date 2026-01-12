# Multi-stage Dockerfile for SwellSight Wave Analysis System

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development image
FROM base as development

WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ requirements/
RUN pip install -r requirements/development.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Stage 3: Training image
FROM base as training

WORKDIR /app

# Copy requirements
COPY requirements/ requirements/
RUN pip install -r requirements/training.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p data outputs logs checkpoints

EXPOSE 8000

CMD ["python", "scripts/train.py"]

# Stage 4: Inference image (production)
FROM base as inference

WORKDIR /app

# Copy requirements
COPY requirements/ requirements/
RUN pip install -r requirements/inference.txt

# Copy only necessary files for inference
COPY src/ src/
COPY configs/inference.yaml configs/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash swellsight
USER swellsight

# Create directories
RUN mkdir -p outputs logs

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "swellsight.api.server:app", "--host", "0.0.0.0", "--port", "8000"]