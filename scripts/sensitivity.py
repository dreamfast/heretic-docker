#!/usr/bin/env python3
"""
Per-layer NVFP4 sensitivity analysis for dynamic mixed-precision quantization.

Measures how much each nn.Linear layer degrades when quantized to NVFP4
(FP4 E2M1, 16-element blocks). The most sensitive layers can be kept in FP8
instead of FP4 to preserve accuracy — the Unsloth "Dynamic" approach.

Two modes:
  - weight:     measures relative weight quantization error (fast, no forward pass)
  - activation: measures output MSE on calibration data (more accurate, needs GPU)
"""

import torch
import torch.nn.functional as F

_FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_MAX = 6.0
_BLOCK_SIZE = 16
# Hoisted LUT — built once per device/dtype on demand
_FP4_LUT_CACHE: dict[tuple, torch.Tensor] = {}


def _get_fp4_lut(device, dtype):
    """Get or create the FP4 lookup table for a given device/dtype."""
    key = (device, dtype)
    if key not in _FP4_LUT_CACHE:
        _FP4_LUT_CACHE[key] = torch.tensor(_FP4_GRID, device=device, dtype=dtype)
    return _FP4_LUT_CACHE[key]


def _fp4_quant_dequant(w: torch.Tensor, block_size: int = _BLOCK_SIZE) -> torch.Tensor:
    """Approximate NVFP4 quantize -> dequantize for sensitivity ranking.

    Uses per-block absmax / 6.0 scaling (the FP4 E2M1 max) and rounds to the
    nearest representable FP4 E2M1 value. This is a proxy — the real NVFP4
    encoder uses FP8 E4M3 block scales, but per-block absmax is close enough
    for ranking layers by sensitivity.
    """
    orig_shape = w.shape
    if w.ndim == 1:
        w = w.unsqueeze(0)

    rows, cols = w.shape
    pad = (block_size - cols % block_size) % block_size
    if pad:
        w = F.pad(w, (0, pad))

    padded_cols = w.shape[1]
    num_blocks = padded_cols // block_size
    w_blk = w.reshape(rows, num_blocks, block_size)

    amax = w_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / _FP4_MAX
    w_n = (w_blk / scale).clamp(-_FP4_MAX, _FP4_MAX)

    fp4_lut = _get_fp4_lut(w.device, w.dtype)
    abs_val = w_n.abs()
    idx = (abs_val.unsqueeze(-1) - fp4_lut).abs().argmin(dim=-1)
    w_q = w_n.sign() * fp4_lut[idx]

    w_dq = (w_q * scale).reshape(rows, padded_cols)
    if pad:
        w_dq = w_dq[:, :cols]
    return w_dq.reshape(orig_shape)


def _iter_linear_layers(model):
    """Yield (name, module) for every nn.Linear except lm_head."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name.lower():
            yield name, module


def weight_sensitivity(model) -> dict[str, float]:
    """Fast weight-only sensitivity. No forward pass, no calibration data.

    Returns relative weight quantization error per layer:
        ||W - W_fp4||_F^2 / ||W||_F^2
    """
    errors = {}
    model_device = next(model.parameters()).device
    for name, module in _iter_linear_layers(model):
        with torch.no_grad():
            w = module.weight.data.to(device=model_device, dtype=torch.float32)
            w_dq = _fp4_quant_dequant(w)
            err = (w - w_dq).pow(2).mean()
            norm = w.pow(2).mean().clamp(min=1e-10)
            errors[name] = (err / norm).item()
    return errors


def activation_sensitivity(
    model, input_ids_batches: list[torch.Tensor], max_batches: int = 8
) -> dict[str, float]:
    """Activation-aware sensitivity. Runs one forward pass over calibration data.

    Measures output MSE: ||y_ref - y_fp4||^2 where y_ref is the layer's BF16
    output and y_fp4 is computed with FP4-quantized weights.

    Args:
        model: the loaded model (eval mode).
        input_ids_batches: list of input_ids tensors (1, seq_len) or (batch, seq_len).
        max_batches: cap on number of batches to process.
    """
    errors: dict[str, float] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            x = inp[0]
            w = module.weight.data
            w_dq = _fp4_quant_dequant(w.to(torch.float32))
            bias = module.bias.to(torch.float32) if module.bias is not None else None
            y_q = F.linear(x.to(torch.float32), w_dq, bias=bias)
            err = F.mse_loss(out.to(torch.float32), y_q, reduction="mean").item()
            errors[layer_name] = errors.get(layer_name, 0.0) + err
            counts[layer_name] = counts.get(layer_name, 0) + 1

        return hook

    for name, module in _iter_linear_layers(model):
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for ids in input_ids_batches[:max_batches]:
            ids = ids.to(model.device)
            try:
                model(ids)
            except RuntimeError:
                break

    for h in hooks:
        h.remove()

    return {name: errors[name] / max(counts[name], 1) for name in errors}


def select_layers(
    errors: dict[str, float], top_k: int | None = None, percentile: float = 0.15
) -> list[str]:
    """Return names of the most sensitive layers.

    Args:
        errors: {layer_name: error_score} from weight_sensitivity or activation_sensitivity.
        top_k: if set, keep exactly this many layers. Overrides percentile.
        percentile: fraction of total layers to keep (default 15%).
    """
    ranked = sorted(errors.items(), key=lambda x: -x[1])
    if top_k and top_k > 0:
        n = min(top_k, len(ranked))
    else:
        n = max(1, int(len(ranked) * percentile))
    selected = [name for name, _ in ranked[:n]]
    return selected


def format_report(errors: dict[str, float], selected: list[str]) -> str:
    """Human-readable sensitivity report."""
    lines = ["", "=" * 60, "  NVFP4 Sensitivity Analysis", "=" * 60, ""]

    ranked = sorted(errors.items(), key=lambda x: -x[1])
    total = len(ranked)

    if total == 0:
        lines.append("  No Linear layers found for analysis.")
        lines.append("=" * 60 + "")
        return "\n".join(lines)

    fp8_set = set(selected)
    sel_pct = len(selected) / total * 100 if total > 0 else 0
    fp4_pct = (total - len(selected)) / total * 100 if total > 0 else 0
    lines.append(f"  Total Linear layers:  {total}")
    lines.append(f"  Kept in FP8:          {len(selected)} ({sel_pct:.0f}%)")
    lines.append(f"  Quantized to NVFP4:   {total - len(selected)} ({fp4_pct:.0f}%)")
    lines.append("")
    lines.append(f"  {'RANK':<5} {'ERROR':>10}  {'FORMAT':<6}  LAYER")
    lines.append(f"  {'----':<5} {'-----':>10}  {'------':<6}  -----")

    for i, (name, err) in enumerate(ranked):
        fmt = "FP8" if name in fp8_set else "FP4"
        marker = " *" if name in fp8_set else ""
        short = name if len(name) <= 50 else "..." + name[-47:]
        lines.append(f"  {i+1:<5} {err:>10.6f}  {fmt:<6}  {short}{marker}")

    lines.append("")
    lines.append("  (* = kept in FP8, rest quantized to NVFP4)")
    lines.append("=" * 60 + "")
    return "\n".join(lines)
