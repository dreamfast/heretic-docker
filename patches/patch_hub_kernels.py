#!/usr/bin/env python3
"""Patch transformers hub_kernels.py to handle kernels package failures gracefully.

The kernels package is for loading custom CUDA kernels from the HuggingFace Hub,
which is unnecessary for heretic/abliteration. It fails at module init due to
missing LayerRepository config. This patch wraps it in try/except.
"""

import sys
from pathlib import Path


def find_target():
    for candidate in sys.path:
        p = Path(candidate) / "transformers" / "integrations" / "hub_kernels.py"
        if p.exists():
            return p
    return None


target = find_target()
if target is None:
    print("transformers/integrations/hub_kernels.py not found, skipping patch")
    sys.exit(0)

code = target.read_text()

if "_KERNELS_STUBBED" in code:
    print("Already patched, skipping")
    sys.exit(0)

stub = '''# _KERNELS_STUBBED
def _noop(*a, **kw): return None
def _decorator(*a, **kw): return lambda f: f
def _bool(*a, **kw): return False
class LayerRepository:
    def __init__(self, *a, **kw): pass
is_kernel = _bool
load_and_register_kernel = _noop
register_kernel = _decorator
lazy_load_kernel = _decorator
get_kernel = lambda *a, **kw: _noop
use_kernel_forward_from_hub = _decorator
use_kernel_func_from_hub = _decorator
replace_kernel_forward_from_hub = _decorator
register_kernel_mapping = _noop
register_kernel_mapping_transformers = _noop
use_kernelized_func = _decorator
allow_all_hub_kernels = _noop
def __getattr__(name):
    return _noop
'''

if "from kernels import" in code:
    lines = code.split("\n")
    new_lines = ['try:']
    for line in lines:
        new_lines.append("    " + line)
    new_lines.append("except Exception:")
    for stub_line in stub.rstrip().split("\n"):
        new_lines.append("    " + stub_line)
    code = "\n".join(new_lines)
    target.write_text(code)
    print(f"Patched {target} with kernels stub fallback")
else:
    print("No 'from kernels import' found, skipping patch")
    sys.exit(0)
