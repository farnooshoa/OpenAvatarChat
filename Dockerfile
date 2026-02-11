# Dockerfile for HuggingFace Spaces
# OpenAvatarChat with Speech Recognition Enhancements
# Optimized for fast builds and reliable deployment

FROM nvidia/cuda:12.4.0-runtime-ubuntu24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install and configure git-lfs
RUN git lfs install

# Install uv package manager for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy only necessary files first (leverage Docker cache)
COPY pyproject.toml setup.cfg ./
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copy source code
COPY src/ ./src/

# Copy models directory structure
COPY models/ ./models/

# Copy resource files
COPY resource/ ./resource/

# Initialize git submodules (if needed)
RUN if [ -f .gitmodules ]; then \
        git submodule update --init --recursive --depth 1; \
    fi

# Create Python virtual environment
RUN uv venv --python 3.11.11

# Upgrade pip and install build tools
RUN uv pip install --upgrade pip setuptools wheel

# Install Python dependencies for the enhanced configuration
# This will use the config file to determine which dependencies to install
RUN uv run install.py --uv --config config/chat_with_groq_enhanced.yaml || \
    uv pip install -e .

# Run post-installation configuration script
RUN chmod +x scripts/post_config_install.sh && \
    ./scripts/post_config_install.sh --config config/chat_with_groq_enhanced.yaml || true

# Download LiteAvatar models
RUN chmod +x scripts/download_liteavatar_weights.sh && \
    bash scripts/download_liteavatar_weights.sh || \
    echo "Model download will occur at runtime"

# Create directories for runtime
RUN mkdir -p /app/logs /app/cache /app/tmp

# Set environment variables for HuggingFace Spaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose HuggingFace Spaces port
EXPOSE 7860

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set the entrypoint
ENTRYPOINT ["uv", "run"]

# Default command - run the demo with enhanced configuration
CMD ["src/demo.py", "--config", "config/chat_with_groq_enhanced.yaml"]
