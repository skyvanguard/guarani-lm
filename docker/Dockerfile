# GuaraniLM v3 training image — Blackwell (RTX 50-series, SM_120) support
# Requires CUDA 12.8 + PyTorch >= 2.7 + Unsloth latest.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git build-essential ca-certificates curl \
        cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# PyTorch built for CUDA 12.8 — needed for Blackwell SM_120
RUN pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Unsloth latest (Blackwell support, pulls compatible transformers/peft/trl/datasets/accelerate/bitsandbytes)
RUN pip install --upgrade unsloth unsloth-zoo

# Auxiliary deps for our pipeline
RUN pip install \
        sentencepiece \
        protobuf \
        hf_transfer \
        pyyaml \
        sacrebleu \
        scikit-learn \
        tqdm

WORKDIR /workspace

CMD ["bash"]
