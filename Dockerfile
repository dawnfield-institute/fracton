# Multi-stage Dockerfile for Fracton
# Supports both CPU and GPU deployments

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.11

# ============================================================================
# Stage 1: Base Builder (common for CPU and GPU)
# ============================================================================
FROM python:${PYTHON_VERSION}-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ============================================================================
# Stage 2: CPU Runtime
# ============================================================================
FROM python:${PYTHON_VERSION}-slim as runtime-cpu

LABEL maintainer="Dawn Field Institute"
LABEL description="Fracton SDK with PAC/SEC/MED theoretical foundations"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY fracton/ ./fracton/
COPY setup.py ./
COPY README.md ./

# Install Fracton in development mode
RUN pip install -e .

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FRACTON_DATA_DIR=/app/data
ENV FRACTON_CACHE_DIR=/app/cache
ENV FRACTON_LOG_DIR=/app/logs
ENV DEVICE=cpu

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fracton; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import fracton; print('Fracton CPU runtime ready')"]

# ============================================================================
# Stage 3: GPU Runtime
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 as runtime-gpu

LABEL maintainer="Dawn Field Institute"
LABEL description="Fracton SDK with GPU support"
LABEL version="1.0.0"

ARG PYTHON_VERSION=3.11

WORKDIR /app

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip${PYTHON_VERSION} 1

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY fracton/ ./fracton/
COPY setup.py ./
COPY README.md ./

# Install Fracton in development mode
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/cache /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FRACTON_DATA_DIR=/app/data
ENV FRACTON_CACHE_DIR=/app/cache
ENV FRACTON_LOG_DIR=/app/logs
ENV DEVICE=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import fracton; import torch; print(f'Fracton GPU runtime ready. CUDA available: {torch.cuda.is_available()}')"]

# ============================================================================
# Stage 4: Development Environment
# ============================================================================
FROM runtime-cpu as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-benchmark \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython \
    jupyter

# Copy tests
COPY tests/ ./tests/

# Set development environment
ENV FRACTON_ENV=development

CMD ["bash"]
