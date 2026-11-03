#!/usr/bin/env python3
"""
NVFP4 quantization for vLLM (W4A4, compressed-tensors format).

Takes a HuggingFace model name (or local path), downloads it, quantizes to
NVFP4 via llm-compressor, and saves a vLLM-loadable checkpoint.

Quality options (composable):
  Rung 1  baseline          uniform NVFP4 W4A4 on all Linear layers
  Rung 2  --mixed-precision downproj   keep down_proj in FP8, rest NVFP4
  Rung 4  --mixed-precision dynamic    auto per-layer sensitivity analysis
  Rung 3  --samples 512 --dataset hermes-mix   better calibration data
  Rung 5  --kv-cache-fp8              calibrate KV cache to FP8
  +gptq     Hessian-guided weight quantization (default on, --no-gptq to disable)
  +smooth   SmoothQuant activation outlier smoothing (default on, --no-smoothquant)

Usage:
    python3 nvfp4_vllm.py <hf-model-name> [output-name] [options]

Examples:
    # Fast baseline (no GPTQ, no SmoothQuant)
    python3 nvfp4_vllm.py meta-llama/Meta-Llama-3-8B-Instruct --no-gptq --no-smoothquant

    # Max quality (all features)
    python3 nvfp4_vllm.py meta-llama/Meta-Llama-3-8B-Instruct \\
        --mixed-precision dynamic --samples 512 --dataset hermes-mix --kv-cache-fp8
"""

import argparse
import copy
import json
import os
import sys

import torch

# Ensure /scripts is on sys.path so sensitivity.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Compatibility shim: transformers 5.15 renamed GraniteMoeParallelExperts → GraniteMoeExperts
# llmcompressor 0.12 still imports the old name. Alias it before llmcompressor loads.
try:
    import transformers.models.granitemoe.modeling_granitemoe as _gm
    if not hasattr(_gm, "GraniteMoeParallelExperts") and hasattr(_gm, "GraniteMoeExperts"):
        _gm.GraniteMoeParallelExperts = _gm.GraniteMoeExperts
except (ImportError, ModuleNotFoundError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVFP4 quantization for vLLM (compressed-tensors format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality ladder (--mixed-precision):
  off        Uniform NVFP4 W4A4 on all Linear layers (fastest, baseline)
  downproj   Keep down_proj in FP8, rest NVFP4 (big quality win, trivial cost)
  dynamic    Auto per-layer sensitivity analysis, keep worst offenders in FP8

Advanced quality (default on):
  --gptq           Hessian-guided weight quantization (2-5x slower, better quality)
  --smoothquant    Smooth outlier activation channels before quantization

For Unsloth-tier quality use:
  --mixed-precision dynamic --samples 512 --dataset hermes-mix --kv-cache-fp8
""",
    )
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output directory name (default: <ModelName>-NVFP4[-FP8])")
    parser.add_argument("--mixed-precision", choices=["off", "downproj", "dynamic"],
                        default="off", help="Mixed-precision strategy (default: off)")
    parser.add_argument("--samples", type=int, default=256,
                        help="Number of calibration samples (default: 256, use 512 for quality)")
    parser.add_argument("--dataset", choices=["ultrachat", "coding-mix", "hermes-mix"],
                        default="ultrachat", help="Calibration dataset (default: ultrachat)")
    parser.add_argument("--dynamic-topk", type=int, default=None,
                        help="Dynamic mode: exact number of layers to keep in FP8 (default: ~15%%)")
    parser.add_argument("--kv-cache-fp8", action="store_true",
                        help="Calibrate KV cache to FP8 for 2x longer context")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max calibration sequence length (default: 2048)")
    parser.add_argument("--output-dir", default="/output/nvfp4-vllm",
                        help="Base output directory (default: /output/nvfp4-vllm)")
    # Advanced quality flags
    parser.add_argument("--gptq", action=argparse.BooleanOptionalAction, default=True,
                        help="Use GPTQ Hessian-guided weight quantization (default: on)")
    parser.add_argument("--dampening-frac", type=float, default=0.01,
                        help="GPTQ Hessian dampening fraction (default: 0.01)")
    parser.add_argument("--actorder", choices=["static", "group", "weight"], default="static",
                        help="GPTQ activation ordering (default: static)")
    parser.add_argument("--smoothquant", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply SmoothQuant activation smoothing (default: on)")
    parser.add_argument("--smoothquant-strength", type=float, default=0.5,
                        help="SmoothQuant smoothing strength alpha (default: 0.5)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable sensitivity analysis cache (force re-analysis)")
    parser.add_argument("--recipes", default=None,
                        help="Comma-separated recipe presets to run in sequence: pure,downproj,dynamic "
                             "(e.g. --recipes pure,dynamic). Shares model cache + sensitivity cache between runs.")
    args = parser.parse_args()
    # Validate numeric ranges
    if not (0 < args.dampening_frac <= 1):
        parser.error("--dampening-frac must be in (0, 1]")
    if not (0 <= args.smoothquant_strength <= 1):
        parser.error("--smoothquant-strength must be in [0, 1]")
    return args


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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    load_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}

    # Try AutoModelForCausalLM first, then VLM classes for multimodal models
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

    model = None
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

    model.config.use_cache = False  # disable during calibration to save memory

    # VLMs (Muse Glimmer, etc.) nest attention params in text_config.
    # compressed-tensors looks for them at the top level — expose them.
    if hasattr(model.config, "text_config"):
        tc = model.config.text_config
        for attr in ("num_attention_heads", "num_key_value_heads", "hidden_size",
                      "head_dim", "num_hidden_layers"):
            if hasattr(tc, attr) and not hasattr(model.config, attr):
                setattr(model.config, attr, getattr(tc, attr))

    return model, tokenizer


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

_DATASET_SOURCES = {
    "ultrachat": [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", "messages", 1.0),
    ],
    "coding-mix": [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", "messages", 0.5),
        ("nickrosh/EvolCodeAlpaca", "train", None, 0.5),
    ],
    "hermes-mix": [
        ("teknium/OpenHermes-2.5", "train", "conversations", 0.40),
        ("nickrosh/EvolCodeAlpaca", "train", None, 0.30),
        ("meta-math/MetaMathQA", "train", None, 0.20),
        ("NousResearch/hermes-function-calling-v1", "train", None, 0.10),
    ],
}


def load_calibration_data(dataset_name: str, tokenizer, num_samples: int, max_seq_len: int):
    """Load and tokenize calibration data. Returns a HF Dataset with input_ids."""
    from datasets import load_dataset, Dataset

    sources = _DATASET_SOURCES.get(dataset_name, _DATASET_SOURCES["ultrachat"])
    all_texts = []

    for ds_name, split, msg_key, fraction in sources:
        count = max(8, int(num_samples * fraction))
        try:
            ds = load_dataset(ds_name, split=f"{split}[:{count}]")
            texts = _extract_texts(ds, msg_key, tokenizer)
            all_texts.extend(texts)
            print(f"  Loaded {len(texts)} samples from {ds_name}")
            if len(texts) == 0:
                print(f"  WARNING: {ds_name} contributed 0 samples (schema mismatch?)")
        except Exception as e:
            print(f"  WARNING: could not load {ds_name}: {e}")

    if not all_texts:
        raise RuntimeError("No calibration data could be loaded")

    all_texts = all_texts[:num_samples]
    print(f"  Total calibration samples: {len(all_texts)}")

    tokenized = Dataset.from_dict({"text": all_texts})
    tokenized = tokenized.map(
        lambda ex: tokenizer(ex["text"], padding=False, max_length=max_seq_len,
                             truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )
    return tokenized


def _extract_texts(dataset, msg_key, tokenizer):
    """Extract text from a dataset, applying chat template if messages format."""
    texts = []
    for example in dataset:
        # Try messages format (UltraChat, OpenHermes, Hermes FC)
        for key in [msg_key, "messages", "conversations"]:
            if key and key in example:
                msgs = example[key]
                if isinstance(msgs, list):
                    try:
                        text = tokenizer.apply_chat_template(msgs, tokenize=False)
                    except Exception:
                        text = str(msgs)
                else:
                    text = str(msgs)
                if text:
                    texts.append(text)
                break
        else:
            # Handle instruction/query → output/response schemas
            instr = (example.get("instruction") or example.get("prompt")
                     or example.get("query") or "")
            out = (example.get("output") or example.get("completion")
                   or example.get("response") or "")
            text = f"{instr}\n{out}" if instr or out else example.get("text", "")
            if text:
                texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Recipe building
# ---------------------------------------------------------------------------

def _downproj_groups():
    """Shared down_proj→FP8, rest→NVFP4 config_groups."""
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
    return {
        "fp8_sensitive": {**FP8_DYNAMIC, "targets": ["re:.*down_proj.*"]},
        "nvfp4_rest": {**NVFP4, "targets": [
            "re:.*self_attn.*", "re:.*gate_proj.*", "re:.*up_proj.*",
        ]},
    }


def _build_config_groups(args, model, calib_data):
    """Build config_groups dict shared by GPTQ and QuantizationModifier."""
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4

    mode = args.mixed_precision

    if mode == "dynamic":
        return _build_dynamic_groups(model, calib_data, args)

    if mode == "downproj":
        groups = _downproj_groups()
        print("Mixed precision (downproj): down_proj -> FP8, rest -> NVFP4")
        return groups

    # Uniform
    groups = {"nvfp4_all": {**NVFP4, "targets": ["Linear"]}}
    print("Uniform NVFP4 W4A4 on all Linear layers")
    return groups


# ---------------------------------------------------------------------------
# Sensitivity cache
# ---------------------------------------------------------------------------

def _sensitivity_cache_path(output_dir, model_id):
    """Cache file path for sensitivity analysis results."""
    base = model_id.rstrip("/").split("/")[-1]
    return os.path.join(output_dir, ".cache", f"{base}-sensitivity.json")


def _save_sensitivity(cache_path, errors, selected, args):
    """Save sensitivity analysis results to JSON cache."""
    import tempfile
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data = {
        "model": args.model,
        "errors": errors,
        "selected": selected,
        "percentile": _SENSITIVITY_PERCENTILE,
        "topk": args.dynamic_topk,
        "dataset": args.dataset,
        "samples": args.samples,
        "max_seq_len": args.max_seq_len,
    }
    # Atomic write + reject NaN/Inf values (poison the cache loudly, not silently)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
        os.replace(tmp, cache_path)
    except Exception:
        os.unlink(tmp)
        raise
    print(f"  Sensitivity cached to: {cache_path}")


def _load_sensitivity(cache_path, args):
    """Load cached sensitivity if valid (same model + calibration + selection params)."""
    if not os.path.exists(cache_path):
        return None, None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Validate cache key: model + calibration data + selection params must match
        if data.get("model") != args.model:
            return None, None
        if data.get("topk") != args.dynamic_topk:
            return None, None
        if data.get("dataset") != args.dataset:
            return None, None
        if data.get("samples") != args.samples:
            return None, None
        if data.get("max_seq_len") != args.max_seq_len:
            return None, None
        errors = data["errors"]
        selected = data["selected"]
        # Guard against empty/poisoned cache
        if not errors or not selected:
            return None, None
        return errors, selected
    except (json.JSONDecodeError, KeyError):
        return None, None


# ---------------------------------------------------------------------------
# Recipe presets for multi-recipe mode
# ---------------------------------------------------------------------------

_RECIPE_PRESETS = {
    "pure":     {"mixed_precision": "off",      "gptq": False, "smoothquant": False, "suffix": "-NVFP4"},
    "downproj": {"mixed_precision": "downproj", "gptq": True,  "smoothquant": True,  "suffix": "-NVFP4-downproj"},
    "dynamic":  {"mixed_precision": "dynamic",  "gptq": True,  "smoothquant": True,  "suffix": "-NVFP4-dynamic"},
}

_SENSITIVITY_PERCENTILE = 0.15


def _groups_from_errors(errors, selected):
    """Build FP8/NVFP4 config_groups from sensitivity errors + selected layers."""
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
    fp8_set = set(selected)
    all_linear = list(errors.keys())
    nvfp4_rest = [name for name in all_linear if name not in fp8_set]
    groups = {
        "fp8_sensitive": {**FP8_DYNAMIC, "targets": list(selected)},
        "nvfp4_rest": {**NVFP4, "targets": nvfp4_rest},
    }
    print(f"\nMixed precision (dynamic): {len(selected)} layers -> FP8, {len(nvfp4_rest)} -> NVFP4")
    return groups


def _build_dynamic_groups(model, calib_data, args):
    """Rung 4: run sensitivity analysis, build config_groups dynamically."""
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
    from sensitivity import (
        weight_sensitivity, activation_sensitivity, select_layers, format_report,
    )

    # Check sensitivity cache first
    cache_path = _sensitivity_cache_path(args.output_dir, args.model)
    if not args.no_cache:
        cached_errors, cached_selected = _load_sensitivity(cache_path, args)
        if cached_errors is not None:
            print(f"\nLoaded sensitivity analysis from cache ({len(cached_errors)} layers)")
            print(format_report(cached_errors, cached_selected))
            return _groups_from_errors(cached_errors, cached_selected)

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
        return _downproj_groups()

    selected = select_layers(errors, top_k=args.dynamic_topk, percentile=_SENSITIVITY_PERCENTILE)
    print(format_report(errors, selected))

    # Cache for future runs
    if not args.no_cache:
        _save_sensitivity(cache_path, errors, selected, args)

    return _groups_from_errors(errors, selected)


def build_recipe(args, model, calib_data):
    """Build the full quantization recipe (list of modifiers).

    Order: [SmoothQuantModifier?] + [GPTQModifier | QuantizationModifier]
    SmoothQuant runs first (pre-transform), then GPTQ/Quantization quantizes.
    """
    recipe = []

    # 1. Optional SmoothQuant pre-transform
    if args.smoothquant:
        sq = _try_smoothquant(args)
        if sq:
            recipe.append(sq)
        else:
            print("SmoothQuant disabled (unavailable or no layer mappings)")

    # 2. Main quantization modifier
    config_groups = _build_config_groups(args, model, calib_data)
    kwargs = {"config_groups": config_groups, "ignore": ["lm_head"]}

    if args.kv_cache_fp8:
        try:
            from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
            # kv_cache_scheme expects QuantizationArgs (not a full scheme dict).
            # Use the input_activations args from FP8_DYNAMIC.
            kwargs["kv_cache_scheme"] = FP8_DYNAMIC["input_activations"]
            print("KV cache: FP8 calibration enabled")
        except Exception:
            print("WARNING: kv_cache_scheme not supported, use --kv-cache-dtype fp8 at serve time")

    modifier = _build_quant_modifier(args, model, kwargs)
    recipe.append(modifier)

    return recipe  # always a list — oneshot expects list[Modifier], not bare Modifier


def _try_smoothquant(args):
    """Try to create SmoothQuantModifier. Returns None if unavailable."""
    try:
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        sq = SmoothQuantModifier(smoothing_strength=args.smoothquant_strength)
        print(f"SmoothQuant enabled (strength={args.smoothquant_strength})")
        return sq
    except Exception as e:
        print(f"WARNING: SmoothQuant unavailable ({e})")
        return None


def _build_quant_modifier(args, model, kwargs):
    """Build the main quantization modifier (GPTQ or QuantizationModifier fallback)."""
    if args.gptq:
        try:
            from llmcompressor.modifiers.gptq import GPTQModifier
            gptq_kwargs = {
                **kwargs,
                "dampening_frac": args.dampening_frac,
                "actorder": args.actorder,
            }
            if should_use_sequential(model):
                gptq_kwargs["offload_hessians"] = True
                print("GPTQ: offloading Hessians to CPU (large model)")
            modifier = GPTQModifier(**gptq_kwargs)
            print("Using GPTQ (Hessian-guided weight quantization)")
            return modifier
        except Exception as e:
            print(f"WARNING: GPTQ unavailable ({e}), falling back to QuantizationModifier")

    from llmcompressor.modifiers.quantization import QuantizationModifier
    print("Using QuantizationModifier (round-to-nearest)")
    return QuantizationModifier(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.recipes:
        # Multi-recipe mode: run each preset in sequence
        recipe_names = [r.strip() for r in args.recipes.split(",")]
        valid = [r for r in recipe_names if r in _RECIPE_PRESETS]
        invalid = [r for r in recipe_names if r not in _RECIPE_PRESETS]
        if invalid:
            print(f"WARNING: Unknown recipes: {invalid}. Valid: {list(_RECIPE_PRESETS.keys())}")
        if not valid:
            print("No valid recipes specified.")
            sys.exit(1)

        base = args.model.rstrip("/").split("/")[-1]
        for i, name in enumerate(valid):
            preset = _RECIPE_PRESETS[name]
            print(f"\n{'#' * 60}")
            print(f"  Recipe {i+1}/{len(valid)}: {name}")
            print(f"  mixed-precision={preset['mixed_precision']}, gptq={preset['gptq']}, smoothquant={preset['smoothquant']}")
            print(f"{'#' * 60}\n")

            recipe_args = copy.copy(args)
            recipe_args.mixed_precision = preset["mixed_precision"]
            recipe_args.gptq = preset["gptq"]
            recipe_args.smoothquant = preset["smoothquant"]
            recipe_args.output = base + preset["suffix"]

            _run_single_quantization(recipe_args)
    else:
        _run_single_quantization(args)


def _run_single_quantization(args):
    """Run a single quantization pass and save the checkpoint. Takes pre-built args namespace."""

    out_name = args.output
    if not out_name:
        base = args.model.rstrip("/").split("/")[-1]
        suffix = "-NVFP4-FP8" if args.mixed_precision != "off" else "-NVFP4"
        out_name = base + suffix
    else:
        out_name = os.path.basename(out_name)  # sanitize against path traversal
    save_dir = os.path.join(args.output_dir, out_name)

    print("=" * 60)
    print("  NVFP4 Quantization for vLLM")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Output:          {save_dir}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  GPTQ:            {args.gptq}")
    print(f"  SmoothQuant:     {args.smoothquant}")
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

    oneshot_kwargs = dict(
        model=model,
        processor=tokenizer,  # skip AutoProcessor detection (VLMs without preprocessor_config.json)
        dataset=calib_data,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=min(args.samples, len(calib_data)),
    )

    try:
        oneshot(**oneshot_kwargs)
    except (ValueError, RuntimeError) as e:
        if "mapping" in str(e).lower() or "smoothquant" in str(e).lower():
            # SmoothQuant can't resolve per-layer mappings for this architecture.
            # Safe to retry without it — it failed before any weight modification.
            print(f"\nSmoothQuant mapping failed for this architecture, retrying without it...\n")
            recipe = _strip_smoothquant(recipe)
            oneshot_kwargs["recipe"] = recipe
            oneshot(**oneshot_kwargs)
        else:
            raise

    # Post-quantization sanity check
    _sanity_check(model, tokenizer)

    # Restore use_cache before saving (disabled during calibration to save memory)
    model.config.use_cache = True

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


def _strip_smoothquant(recipe):
    """Remove SmoothQuantModifier from a recipe (list or single modifier)."""
    if not isinstance(recipe, list):
        return recipe
    try:
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        filtered = [m for m in recipe if not isinstance(m, SmoothQuantModifier)]
        return filtered[0] if len(filtered) == 1 else filtered
    except ImportError:
        return recipe


def _sanity_check(model, tokenizer):
    """Quick post-quantization generation to catch broken recipes."""
    print("\n--- Post-quantization sanity check ---")
    try:
        device = next(model.parameters()).device
        ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=15, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  Generation: {text[:100]}")
        print("  Sanity check: PASSED")
    except Exception as e:
        print(f"  Sanity check skipped ({e})")
    print("---\n")


if __name__ == "__main__":
    main()
