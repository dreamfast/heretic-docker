# Heretic LLM Abliteration Pipeline
# NGC PyTorch base provides Blackwell (sm_120) CUDA kernel support.
# Also works on older GPUs (Ada, Ampere, etc).

FROM nvcr.io/nvidia/pytorch:26.10-py3

# gosu for UID/GID matching at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Nuke ALL conflicting NGC packages cleanly, then install heretic (v2).
# Keep torch + triton + torchvision from NGC (the sm_120 builds we need).
# torchvision is required by multimodal/VL models (e.g. Qwen3-VL) whose
# AutoProcessor loads an AutoVideoProcessor; keep the NGC-matched build to
# avoid a CUDA/version mismatch with the NGC torch.
#
# heretic v2 has native headless operation (--checkpoint-action, --trial-index,
# --model-action, --save-directory, --export-strategy) which enables automated
# sweep workflows. heretic v2 pulls transformers[kernels]~=5.18 and all other
# deps itself.
RUN pip uninstall -y \
    huggingface-hub transformers tokenizers accelerate safetensors \
    datasets peft kernels 2>/dev/null || true && \
    find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \( \
    -name "huggingface_hub*" -o -name "transformers*" -o -name "tokenizers*" \
    -o -name "accelerate*" -o -name "safetensors*" -o -name "datasets*" \
    -o -name "peft*" -o -name "kernels*" \) | xargs rm -rf && \
    git clone --branch master --depth 1 https://github.com/p-e-w/heretic.git /tmp/heretic && \
    pip install --no-cache-dir /tmp/heretic && \
    rm -rf /tmp/heretic

# NOTE: huggingface_hub >= 1.33 fixed the PEP 604 union-type validation that
# patch_hf_union_types.py used to work around -- that patch is no longer needed.
#
# Patch transformers:
# - hub_kernels: kernels package init failures break module import (still needed on 5.22)
# - tokenizer special tokens: Gemma-4 style extra_special_tokens passed as list
#   (upstream still dict-only as of transformers 5.22)
# Applied AFTER all pip installs to prevent downgrades from undoing the patches.
COPY patches/patch_hub_kernels.py /tmp/patch_hub_kernels.py
COPY patches/patch_tokenizer_special_tokens.py /tmp/patch_tokenizer_special_tokens.py

# SDPA patch disabled -- it breaks heretic abliteration on Qwen3
# Only needed for Blackwell (5090) GPUs. Re-enable if running on 5090.
# COPY patches/patch_sdpa.py /tmp/patch_sdpa.py
# RUN python3 /tmp/patch_sdpa.py \
#     $(find /usr/local/lib/python3.12/dist-packages -path "*/heretic/model.py" -print -quit) && \
#     rm /tmp/patch_sdpa.py

# huggingface_hub 1.x uses Xet for high-performance transfers (hf-xet is bundled);
# the old hf-transfer Rust downloader is deprecated.
ENV HF_XET_HIGH_PERFORMANCE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;90;100;120" && \
    cmake --build build --config Release -j$(nproc) --target llama-quantize llama-imatrix && \
    # Install only the GGUF conversion deps that llama.cpp actually needs.
    # Do NOT use the requirements file -- it pins torch==2.11.0 (CPU) and
    # transformers==4.57.6, which destroy the NGC torch + heretic's transformers.
    pip install --no-cache-dir "sentencepiece>=0.1.98,<0.3.0" "protobuf>=4.21.0,<5.0.0" && \
    pip install --no-cache-dir --force-reinstall --no-deps ./gguf-py && \
    rm -rf /llama.cpp/.git /llama.cpp/gguf-py

RUN python3 /tmp/patch_hub_kernels.py && \
    python3 /tmp/patch_tokenizer_special_tokens.py && \
    rm /tmp/patch_hub_kernels.py /tmp/patch_tokenizer_special_tokens.py

# Install quantization toolkit: convert-to-quant (INT4 W4A4 ConvRot, INT8 ConvRot, FP8, MXFP8, NVFP4)
# and comfy-kitchen (CUDA/Triton kernels for NVFP4/MXFP8 quantization & dequantization)
# --no-deps prevents pip from replacing the NGC CUDA torch with a CPU-only PyPI wheel
# The patch below adds INT4 W4A4 ConvRot support (PR silveroxides/convert_to_quant#55) on top of
# the PyPI release. The guard skips patching once upstream ships INT4 natively (forward-compatible).
COPY patches/convert_to_quant_int4.patch /tmp/convert_to_quant_int4.patch
RUN pip install --no-cache-dir --no-deps convert-to-quant && \
    if ! python3 -c "from convert_to_quant.constants import INT4_SYMMETRIC_MAX" 2>/dev/null; then \
        echo "Patching convert-to-quant: adding INT4 W4A4 ConvRot (PR #55)"; \
        CTQ_DIR=$(python3 -c "import convert_to_quant, os; print(os.path.dirname(convert_to_quant.__file__))"); \
        patch --no-backup-if-mismatch -p2 -d "$CTQ_DIR" < /tmp/convert_to_quant_int4.patch; \
    else \
        echo "convert-to-quant already has INT4; skipping patch"; \
    fi && \
    rm -f /tmp/convert_to_quant_int4.patch && \
    pip install --no-cache-dir --no-deps prodigy-plus-schedule-free && \
    (pip install --no-cache-dir --no-deps "comfy-kitchen[cublas]" 2>/dev/null || \
     pip install --no-cache-dir --no-deps comfy-kitchen 2>/dev/null || \
     echo "WARNING: comfy-kitchen not installed, MXFP8/NVFP4 CUDA kernels unavailable")

# Install llm-compressor for vLLM NVFP4 quantization (compressed-tensors format).
# heretic v2 already pulls transformers ~=5.18; this floor ensures we still get
# the latest (5.22+) for newer architectures (Muse Glimmer, etc).
RUN pip install --no-cache-dir "transformers>=5.18.0" && \
    pip install --no-cache-dir --no-deps "compressed-tensors==0.20.1" && \
    pip install --no-cache-dir loguru datasets && \
    pip install --no-cache-dir --no-deps "llmcompressor==0.15.0"

# Re-apply the transformers tokenizer special-tokens patch that the upgrade
# above invalidated (hub_kernels patch is version-tolerant, re-apply anyway)
COPY patches/patch_hub_kernels.py /tmp/patch_hub_kernels.py
COPY patches/patch_tokenizer_special_tokens.py /tmp/patch_tokenizer_special_tokens.py
RUN python3 /tmp/patch_hub_kernels.py && \
    python3 /tmp/patch_tokenizer_special_tokens.py && \
    rm /tmp/patch_hub_kernels.py /tmp/patch_tokenizer_special_tokens.py

# Verify torch survived all pip installs (no GPU during build, so check for CPU-only
# build rather than cuda.is_available). The NGC torch reports cuda.is_available()=False
# without hardware, but is NOT a CPU-only build.
RUN python3 -c "import torch; v=torch.__version__; assert '+cpu' not in v, f'CPU-only torch detected: {v}'; print(f'torch OK: {v}')"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
