#!/usr/bin/env python3
"""Patch transformers tokenizer to handle extra_special_tokens as list or dict.

Gemma 4 passes extra_special_tokens as a list (e.g. ["<image>", "<video>"]) but
_set_model_specific_special_tokens expects a dict with .keys().
"""

import sys
from pathlib import Path


def find_target():
    for candidate in sys.path:
        p = Path(candidate) / "transformers" / "tokenization_utils_base.py"
        if p.exists():
            return p
    return None


target = find_target()
if target is None:
    print("transformers/tokenization_utils_base.py not found, skipping patch")
    sys.exit(0)

code = target.read_text()

if "_EXTRA_TOKENS_PATCHED" in code:
    print("Already patched, skipping")
    sys.exit(0)

old = 'self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens'

if old not in code:
    print("Target pattern not found, skipping patch")
    sys.exit(0)

patch_code = '''if isinstance(special_tokens, list):
            special_tokens = {t: t if isinstance(t, str) else t.content for t in special_tokens}  # _EXTRA_TOKENS_PATCHED
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens'''

code = code.replace(old, patch_code, 1)
target.write_text(code)
print(f"Patched {target} to handle extra_special_tokens as list")
