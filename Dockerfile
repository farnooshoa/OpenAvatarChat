FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy application files
COPY . .

# Create models directory
RUN mkdir -p models

# Install core dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install fastrtc from local wheel
RUN pip install --no-cache-dir libs/fastrtc-0.0.28.dev0-py3-none-any.whl

# Install all other required packages
RUN pip install --no-cache-dir \
    gradio \
    huggingface_hub \
    numpy \
    pyyaml \
    pydantic \
    loguru \
    omegaconf \
    dynaconf \
    funasr \
    modelscope \
    soundfile \
    librosa \
    edge-tts \
    openai \
    python-dotenv \
    Pillow \
    av \
    requests \
    scipy \
    scikit-learn \
    transformers \
    sentencepiece \
    protobuf

# Expose Gradio port
EXPOSE 7860

# Run the demo
CMD ["python", "src/demo.py", "--config", "config/chat_with_groq_enhanced.yaml"]
