FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip --version

RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir faiss-cpu
RUN pip3 install --no-cache-dir sentence-transformers
RUN pip3 install --no-cache-dir "dvc[s3]"

WORKDIR /workspace
COPY . /workspace

RUN python3 -c "from sentence_transformers import SentenceTransformer; import os; print('Downloading sentence-transformers/all-MiniLM-L6-v2...'); model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('/workspace/all-MiniLM-L6-v2'); print('Model successfully downloaded and saved to /workspace')"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DISABLE_TELEMETRY=1

ENV MODEL_PATH="/workspace/yandexgpt-5-lite-8b-instruct-q4_k_m.gguf"

RUN echo "=== DOCKER IMAGE SIZE CHECK ===" && \
    du -sh /workspace

ENTRYPOINT ["python3", "solution.py"]