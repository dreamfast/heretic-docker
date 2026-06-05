# Heretic LLM Abliteration Pipeline
# NGC PyTorch base provides Blackwell (sm_120) CUDA kernel support.
# Also works on older GPUs (Ada, Ampere, etc).

FROM nvcr.io/nvidia/pytorch:26.02-py3

# gosu for UID/GID matching at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Nuke ALL conflicting NGC packages cleanly, then install heretic v1.3.0.
# Keep only torch + triton from NGC (the sm_120 builds we need).
RUN pip uninstall -y \
    huggingface-hub transformers tokenizers accelerate safetensors \
    datasets peft 2>/dev/null || true && \
    find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \
    -name "huggingface_hub*" -o -name "transformers*" -o -name "tokenizers*" \
    -o -name "accelerate*" -o -name "safetensors*" -o -name "datasets*" \
    -o -name "peft*" | xargs rm -rf && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git && \
    git clone --branch v1.3.0 --depth 1 https://github.com/p-e-w/heretic.git /tmp/heretic && \
    pip install --no-cache-dir /tmp/heretic hf-transfer && \
    rm -rf /tmp/heretic

# SDPA patch disabled -- it breaks heretic abliteration on Qwen3
# Only needed for Blackwell (5090) GPUs. Re-enable if running on 5090.
# COPY patches/patch_sdpa.py /tmp/patch_sdpa.py
# RUN python3 /tmp/patch_sdpa.py \
#     $(find /usr/local/lib/python3.12/dist-packages -path "*/heretic/model.py" -print -quit) && \
#     rm /tmp/patch_sdpa.py

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && \
    cmake -B build -DLLAMA_CUDA=OFF -DGGML_CUDA=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-quantize && \
    pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt && \
    rm -rf /llama.cpp/.git /llama.cpp/gguf-py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
