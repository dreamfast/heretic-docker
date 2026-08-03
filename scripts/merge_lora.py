#!/usr/bin/env python3
"""Merge a heretic LoRA adapter into the base model.

Usage:
    python3 merge_lora.py <base_model_id_or_path> <adapter_dir> <output_dir>

The adapter_dir must contain adapter_config.json and adapter_model.safetensors
(produced by heretic --export-strategy adapter).
"""

import sys
from pathlib import Path


def merge_lora(base_model: str, adapter_path: str, output_dir: str):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    model = _load_model(base_model)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("Done!")


def _load_model(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # Try text-only first, fall back to VL model class
    try:
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError:
        print("Text-only model class failed, trying VL model class...")
        return AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <base_model> <adapter_dir> <output_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    merge_lora(sys.argv[1], sys.argv[2], sys.argv[3])
