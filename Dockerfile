# Base stage with common dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create and set working directory
WORKDIR /app

# Builder/test stage
FROM base as builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage caching
COPY pyproject.toml README.md ./

# Install build and test dependencies
RUN python -m pip install --upgrade pip && \
    pip install build twine && \
    pip install -e ".[dev]"

# Copy the rest of the code
COPY . .

# Production stage
FROM base as production

# Copy only what's needed for production
COPY pyproject.toml README.md ./
COPY code_preprocessor/ ./code_preprocessor/

# Install only production dependencies
RUN pip install --no-cache-dir .
