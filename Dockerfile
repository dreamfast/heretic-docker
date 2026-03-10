# Heretic LLM Abliteration + Conversion Pipeline
# NGC PyTorch base provides Blackwell (sm_120) CUDA kernel support.
# Also works on older GPUs (Ada, Ampere, etc).

FROM nvcr.io/nvidia/pytorch:26.02-py3

# gosu for UID/GID matching at runtime, cmake for llama.cpp build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Nuke ALL conflicting NGC packages cleanly, then install heretic fresh.
# Keep only torch + triton from NGC (the sm_120 builds we need).
RUN pip uninstall -y \
    huggingface-hub transformers tokenizers accelerate safetensors \
    datasets peft bitsandbytes 2>/dev/null || true && \
    find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
    -name "huggingface_hub*" -o -name "transformers*" -o -name "tokenizers*" \
    -o -name "accelerate*" -o -name "safetensors*" -o -name "datasets*" \
    -o -name "peft*" -o -name "bitsandbytes*" | xargs rm -rf && \
    pip install --no-cache-dir heretic-llm hf-transfer comfy-kitchen gguf sentencepiece

# Stub bitsandbytes (heretic imports it but no CUDA 13.1 binary exists)
COPY patches/blackwell_compat.py /usr/local/lib/python3.12/dist-packages/heretic_blackwell_compat.py
RUN echo "import heretic_blackwell_compat" > \
    $(python3 -c "import site; print(site.getsitepackages()[0])")/heretic-blackwell.pth

# Patch heretic's model.py to force SDPA attention (Blackwell compatible)
COPY patches/patch_sdpa.py /tmp/patch_sdpa.py
RUN python3 /tmp/patch_sdpa.py \
    $(find /usr/local/lib/python3.12/dist-packages -path "*/heretic/model.py" -print -quit) && \
    rm /tmp/patch_sdpa.py

# llama.cpp for GGUF conversion and quantization
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cmake -B /llama.cpp/build -S /llama.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120" && \
    cmake --build /llama.cpp/build --target llama-quantize -j$(nproc) && \
    rm -rf /llama.cpp/.git /llama.cpp/build/CMakeFiles

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY entrypoint.sh /entrypoint.sh
COPY scripts/ /scripts/
RUN chmod +x /entrypoint.sh /scripts/*.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
