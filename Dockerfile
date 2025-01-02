# Base stage with CUDA and Python
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as base

# Install Python and basic dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --no-cache-dir pip setuptools wheel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    DEBIAN_FRONTEND=noninteractive

# Create and set working directory
WORKDIR /app

# Builder stage for compiling dependencies
FROM base as builder

# Install build dependencies and create venv in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3.10-dev \
    && python3.10 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python -m pip install --no-cache-dir pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage caching
COPY pyproject.toml README.md ./

# Install dependencies
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir build twine && \
    pip install --no-cache-dir -e ".[dev]" && \
    # Clean up unnecessary files
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "*.dist-info" -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    find /opt/venv -type f -name "*.pyd" -delete

# Test stage
FROM base as test

# Copy only the necessary files from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the source code
COPY . .

# Production stage - using distroless base for smallest possible image
FROM gcr.io/distroless/python3-debian11 as production

# Copy CUDA libraries from base
COPY --from=base /usr/local/cuda/lib64 /usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Copy only what's needed for production
COPY pyproject.toml README.md ./
COPY code_preprocessor/ ./code_preprocessor/

# Install only production dependencies
COPY --from=builder /opt/venv/lib/python3.10/site-packages /usr/lib/python3.10/site-packages
