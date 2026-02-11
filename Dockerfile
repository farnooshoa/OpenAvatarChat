FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

COPY . .

RUN git submodule update --init --recursive --depth 1

RUN uv venv --python 3.11
RUN uv pip install setuptools pip
RUN uv run install.py --uv --config config/chat_with_groq_enhanced.yaml
RUN chmod +x scripts/post_config_install.sh
RUN ./scripts/post_config_install.sh --config config/chat_with_groq_enhanced.yaml
RUN bash scripts/download_liteavatar_weights.sh

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["uv", "run", "src/demo.py", "--config", "config/chat_with_groq_enhanced.yaml"]