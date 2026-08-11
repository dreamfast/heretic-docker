#!/usr/bin/env python3
"""
NVFP4 quantization for vLLM (W4A4, compressed-tensors format).

Takes a HuggingFace model name (or local path), downloads it, quantizes to
NVFP4 via llm-compressor, and saves a vLLM-loadable checkpoint.

Quality options (composable):
  Rung 1  baseline          uniform NVFP4 W4A4 on all Linear layers
  Rung 2  --mixed-precision downproj   keep down_proj in FP8, rest NVFP4
  Rung 4  --mixed-precision dynamic    auto per-layer sensitivity analysis
  Rung 3  --samples 512 --dataset coding-mix   better calibration data
  Rung 5  --kv-cache-fp8              calibrate KV cache to FP8

Usage:
    python3 nvfp4_vllm.py <hf-model-name> [output-name] [options]

Examples:
    # Fast baseline
    python3 nvfp4_vllm.py meta-llama/Meta-Llama-3-8B-Instruct

    # Max quality (all rungs)
    python3 nvfp4_vllm.py meta-llama/Meta-Llama-3-8B-Instruct \\
        --mixed-precision dynamic --samples 512 --dataset coding-mix --kv-cache-fp8
"""

import argparse
import os
import sys

import torch

# Ensure /scripts is on sys.path so sensitivity.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVFP4 quantization for vLLM (compressed-tensors format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality ladder (--mixed-precision):
  off        Uniform NVFP4 W4A4 on all Linear layers (fastest, baseline)
  downproj   Keep down_proj in FP8, rest NVFP4 (big quality win, trivial cost)
  dynamic    Auto per-layer sensitivity analysis, keep worst offenders in FP8

The flags compose freely. For Unsloth-tier quality use:
  --mixed-precision dynamic --samples 512 --dataset coding-mix --kv-cache-fp8
""",
    )
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output directory name (default: <ModelName>-NVFP4[-FP8])",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["off", "downproj", "dynamic"],
        default="off",
        help="Mixed-precision strategy (default: off = uniform NVFP4)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256, bump to 512 for quality)",
    )
    parser.add_argument(
        "--dataset",
        choices=["ultrachat", "coding-mix"],
        default="ultrachat",
        help="Calibration dataset (default: ultrachat)",
    )
    parser.add_argument(
        "--dynamic-topk",
        type=int,
        default=None,
        help="Dynamic mode: exact number of layers to keep in FP8 (default: ~15%%)",
    )
    parser.add_argument(
        "--kv-cache-fp8",
        action="store_true",
        help="Calibrate KV cache to FP8 for 2x longer context",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max calibration sequence length (default: 2048)",
    )
    parser.add_argument(
        "--output-dir",
        default="/output/nvfp4-vllm",
        help="Base output directory (default: /output/nvfp4-vllm)",
    )
    return parser.parse_args()


def should_use_sequential(model) -> bool:
    """Check if model needs sequential onloading (won't fit in VRAM)."""
    if not torch.cuda.is_available():
        return False
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    return param_bytes * 2 > vram_bytes


def load_model(model_id: str, kv_cache_fp8: bool = False):
    """Load model + tokenizer, with sequential onloading for large models."""
    from transformers import AutoTokenizer, AutoConfig

    print(f"Loading model: {model_id}")

    # trust_remote_code=True for custom architectures (e.g. Muse Glimmer, Glow)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect model type and pick the right Auto class
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    model = None
    load_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}

    # Build list of candidate Auto classes (import-safe)
    auto_classes = []
    try:
        from transformers import AutoModelForCausalLM
        auto_classes.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText
        auto_classes.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    except ImportError:
        pass

    for class_name, auto_class in auto_classes:
        try:
            model = auto_class.from_pretrained(model_id, **load_kwargs)
            print(f"  Loaded via {class_name}")
            break
        except (ValueError, KeyError, TypeError, OSError) as e:
            print(f"  {class_name} failed: {e}")
            continue

    if model is None:
        raise ValueError(f"Could not load model '{model_id}' (model_type={model_type}). "
                         f"Transformers may not support this architecture yet.")

    sequential = should_use_sequential(model)
    if sequential:
        param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Model is {param_gb:.1f} GB, GPU has {vram_gb:.1f} GB VRAM — using sequential onloading")
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    # Keep KV cache enabled when calibrating it; disable otherwise to save memory
    model.config.use_cache = kv_cache_fp8
    return model, tokenizer


def load_calibration_data(dataset_name: str, tokenizer, num_samples: int, max_seq_len: int):
    """Load and tokenize calibration data. Returns a HF Dataset with input_ids."""
    from datasets import load_dataset, Dataset

    sources = []
    if dataset_name == "coding-mix":
        per = num_samples // 2
        sources = [
            ("HuggingFaceH4/ultrachat_200k", "train_sft", "messages", per),
            ("nickrosh/EvolCodeAlpaca", "train", None, num_samples - per),
        ]
    else:
        sources = [("HuggingFaceH4/ultrachat_200k", "train_sft", "messages", num_samples)]

    all_texts = []
    for ds_name, split, msg_key, count in sources:
        try:
            ds = load_dataset(ds_name, split=f"{split}[:{count}]")
            texts = _extract_texts(ds, msg_key, tokenizer)
            all_texts.extend(texts)
            print(f"  Loaded {len(texts)} samples from {ds_name}")
        except Exception as e:
            print(f"  WARNING: could not load {ds_name}: {e}")

    if not all_texts:
        raise RuntimeError("No calibration data could be loaded")

    all_texts = all_texts[:num_samples]
    print(f"  Total calibration samples: {len(all_texts)}")

    tokenized = Dataset.from_dict({"text": all_texts})
    tokenized = tokenized.map(
        lambda ex: tokenizer(
            ex["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=["text"],
    )
    return tokenized


def _extract_texts(dataset, msg_key, tokenizer):
    """Extract text from a dataset, applying chat template if messages format."""
    texts = []
    for example in dataset:
        if msg_key and msg_key in example:
            msgs = example[msg_key]
            if isinstance(msgs, list):
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            else:
                text = str(msgs)
        else:
            # Handle common dataset schemas: instruction/output, prompt/completion, text
            instr = example.get("instruction") or example.get("prompt") or ""
            out = example.get("output") or example.get("completion") or ""
            text = f"{instr}\n{out}" if instr or out else example.get("text", "")
        if text:
            texts.append(text)
    return texts


def build_recipe(args, model, calib_data):
    """Build the QuantizationModifier recipe based on CLI flags."""
    from llmcompressor.modifiers.quantization import QuantizationModifier

    nvfp4_targets = [
        "re:.*self_attn.k_proj.*",
        "re:.*self_attn.o_proj.*",
        "re:.*self_attn.q_proj.*",
        "re:.*self_attn.v_proj.*",
        "re:.*gate_proj.*",
        "re:.*up_proj.*",
        "re:.*down_proj.*",
    ]

    mode = args.mixed_precision
    config_groups = {}

    if mode == "dynamic":
        config_groups = _build_dynamic_groups(model, calib_data, args)
    elif mode == "downproj":
        from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4

        fp8_targets = ["re:.*down_proj.*"]
        nvfp4_dynamic = [t for t in nvfp4_targets if "down_proj" not in t]
        config_groups = {
            "fp8_sensitive": {**FP8_DYNAMIC, "targets": fp8_targets},
            "nvfp4_rest": {**NVFP4, "targets": nvfp4_dynamic},
        }
        print(f"Mixed precision (downproj): down_proj -> FP8, rest -> NVFP4")
    else:
        from compressed_tensors.quantization.quant_scheme import NVFP4

        config_groups = {"nvfp4_all": {**NVFP4, "targets": "Linear"}}
        print("Uniform NVFP4 W4A4 on all Linear layers")

    kwargs = {"config_groups": config_groups, "ignore": ["lm_head"]}

    if args.kv_cache_fp8:
        try:
            from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC

            kwargs["kv_cache_scheme"] = FP8_DYNAMIC
            print("KV cache: FP8 calibration enabled")
        except Exception:
            print("WARNING: kv_cache_scheme not supported in this llmcompressor version")
            print("         Use --kv-cache-dtype fp8 at vLLM serve time instead")

    return QuantizationModifier(**kwargs)


def _build_dynamic_groups(model, calib_data, args):
    """Rung 4: run sensitivity analysis, build config_groups dynamically."""
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
    from sensitivity import (
        weight_sensitivity,
        activation_sensitivity,
        select_layers,
        format_report,
    )

    print("\nRunning NVFP4 sensitivity analysis...")

    use_activation = not should_use_sequential(model) and torch.cuda.is_available()
    if use_activation:
        batches = []
        for i in range(min(8, len(calib_data))):
            ids = calib_data[i]["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor([ids])
            batches.append(ids)
        try:
            errors = activation_sensitivity(model, batches)
        except RuntimeError as e:
            print(f"  Activation analysis failed ({e}), falling back to weight-only")
            errors = weight_sensitivity(model)
    else:
        print("  Using weight-only sensitivity (model too large for forward pass)")
        errors = weight_sensitivity(model)

    if not errors:
        print("  WARNING: sensitivity analysis returned no layers, falling back to downproj heuristic")
        return {
            "fp8_sensitive": {**FP8_DYNAMIC, "targets": ["re:.*down_proj.*"]},
            "nvfp4_rest": {**NVFP4, "targets": [
                "re:.*self_attn.*", "re:.*gate_proj.*", "re:.*up_proj.*",
            ]},
        }

    selected = select_layers(errors, top_k=args.dynamic_topk, percentile=0.15)
    print(format_report(errors, selected))

    # Build a clean partition: FP8 layers as exact names,
    # NVFP4 layers as the complement (all other Linear layers as exact names).
    # This avoids overlapping regex/nexact target conflicts in compressed-tensors.
    fp8_set = set(selected)
    all_linear = [name for name in errors.keys()]
    nvfp4_rest = [name for name in all_linear if name not in fp8_set]

    groups = {
        "fp8_sensitive": {**FP8_DYNAMIC, "targets": list(selected)},
        "nvfp4_rest": {**NVFP4, "targets": nvfp4_rest},
    }
    print(f"\nMixed precision (dynamic): {len(selected)} layers -> FP8, {len(nvfp4_rest)} -> NVFP4")
    return groups


def main():
    args = parse_args()

    out_name = args.output
    if not out_name:
        base = args.model.rstrip("/").split("/")[-1]
        suffix = "-NVFP4-FP8" if args.mixed_precision != "off" else "-NVFP4"
        out_name = base + suffix
    save_dir = os.path.join(args.output_dir, out_name)

    print("=" * 60)
    print("  NVFP4 Quantization for vLLM")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Output:          {save_dir}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Calibration:     {args.dataset} ({args.samples} samples)")
    print(f"  KV cache FP8:    {args.kv_cache_fp8}")
    print(f"  Max seq length:  {args.max_seq_len}")
    print("=" * 60 + "\n")

    model, tokenizer = load_model(args.model, kv_cache_fp8=args.kv_cache_fp8)

    print("Loading calibration data...")
    calib_data = load_calibration_data(args.dataset, tokenizer, args.samples, args.max_seq_len)

    recipe = build_recipe(args, model, calib_data)

    print("\nRunning quantization (oneshot)...\n")
    from llmcompressor import oneshot

    oneshot(
        model=model,
        dataset=calib_data,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=min(args.samples, len(calib_data)),
    )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    print(f"\n{'=' * 60}")
    print(f"  Done! Saved to: {save_dir}")
    print(f"{'=' * 60}")
    print(f"\nServe with vLLM (Blackwell GPU required for W4A4 speedup):")
    print(f"  vllm serve {save_dir}")
    if args.kv_cache_fp8:
        print(f"  vllm serve {save_dir} --kv-cache-dtype fp8")


if __name__ == "__main__":
    main()
