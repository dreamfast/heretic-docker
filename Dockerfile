# Heretic LLM Abliteration Pipeline
# NGC PyTorch base provides Blackwell (sm_120) CUDA kernel support.
# Also works on older GPUs (Ada, Ampere, etc).

FROM nvcr.io/nvidia/pytorch:26.02-py3

# gosu for UID/GID matching at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Nuke ALL conflicting NGC packages cleanly, then install heretic master.
# Keep torch + triton + torchvision from NGC (the sm_120 builds we need).
# torchvision is required by multimodal/VL models (e.g. Qwen3-VL) whose
# AutoProcessor loads an AutoVideoProcessor; keep the NGC-matched build to
# avoid a CUDA/version mismatch with the NGC torch.
#
# Master branch provides native headless mode (--checkpoint-action, --trial-index,
# --model-action, --save-directory) which enables automated sweep workflows.
# heretic master pulls transformers[kernels]~=5.6 and all other deps itself.
RUN pip uninstall -y \
    huggingface-hub transformers tokenizers accelerate safetensors \
    datasets peft kernels 2>/dev/null || true && \
    find /usr/local/lib/python3.12/dist-packages -maxdepth 1 \( \
    -name "huggingface_hub*" -o -name "transformers*" -o -name "tokenizers*" \
    -o -name "accelerate*" -o -name "safetensors*" -o -name "datasets*" \
    -o -name "peft*" -o -name "kernels*" \) | xargs rm -rf && \
    git clone --branch master --depth 1 https://github.com/p-e-w/heretic.git /tmp/heretic && \
    pip install --no-cache-dir /tmp/heretic hf-transfer && \
    rm -rf /tmp/heretic

# Patch huggingface_hub to handle PEP 604 union types (str | None) used by
# the kernels package. Without this, the strict dataclass validator crashes.
# Applied AFTER all pip installs to prevent downgrades from undoing the patch.
COPY patches/patch_hf_union_types.py /tmp/patch_hf_union_types.py
COPY patches/patch_hub_kernels.py /tmp/patch_hub_kernels.py
COPY patches/patch_tokenizer_special_tokens.py /tmp/patch_tokenizer_special_tokens.py
COPY patches/patch_n_top_trials.py /tmp/patch_n_top_trials.py

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
    pip install --no-cache-dir --force-reinstall --no-deps ./gguf-py && \
    rm -rf /llama.cpp/.git /llama.cpp/gguf-py

RUN python3 /tmp/patch_hf_union_types.py && \
    python3 /tmp/patch_hub_kernels.py && \
    python3 /tmp/patch_tokenizer_special_tokens.py && \
    python3 /tmp/patch_n_top_trials.py && \
    rm /tmp/patch_hf_union_types.py /tmp/patch_hub_kernels.py /tmp/patch_tokenizer_special_tokens.py /tmp/patch_n_top_trials.py

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

# Verify CUDA torch survived all pip installs; reinstall from CUDA index if corrupted
RUN python3 -c "import torch; assert torch.cuda.is_available(), f'torch={torch.__version__}'" || \
    (echo "ERROR: CUDA torch corrupted, reinstalling..." && \
     TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
     pip install --no-cache-dir --force-reinstall "torch==${TORCH_VER}" \
       --index-url https://download.pytorch.org/whl/cu130)

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
