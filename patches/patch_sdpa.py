#!/usr/bin/env python3
"""
Patch heretic's model.py to inject attn_implementation='sdpa'
into all from_pretrained calls (except AutoTokenizer).

This inserts the kwarg after **extra_kwargs or after the last keyword arg,
ensuring it comes after all positional arguments.
"""

import re
import sys

model_py = sys.argv[1]

with open(model_py) as f:
    content = f.read()

# Strategy: add attn_implementation="sdpa" to the **extra_kwargs dict
# that heretic passes to from_pretrained. This is the cleanest approach.
#
# In Model.__init__ (line ~99):
#   extra_kwargs = {}
#   if quantization_config is not None:
#       extra_kwargs["quantization_config"] = quantization_config
#
# We add: extra_kwargs["attn_implementation"] = "sdpa"
# after the quantization_config assignment block.
#
# For the other two from_pretrained calls that don't use extra_kwargs,
# we add attn_implementation="sdpa" as a keyword argument before the closing paren.

# Patch 1: The main loading loop - inject into extra_kwargs
# Find: extra_kwargs["quantization_config"] = quantization_config
# Add after: extra_kwargs["attn_implementation"] = "sdpa"
content = content.replace(
    'extra_kwargs["quantization_config"] = quantization_config\n',
    'extra_kwargs["quantization_config"] = quantization_config\n'
    '                    extra_kwargs["attn_implementation"] = "sdpa"\n',
    1  # only first occurrence
)

# Also need to handle the case where quantization_config is None
# (the extra_kwargs dict is empty). Add it unconditionally after extra_kwargs = {}
# Actually, let's just always set it. Replace the extra_kwargs initialization.
# Find the first: extra_kwargs = {}
content = content.replace(
    '                extra_kwargs = {}\n',
    '                extra_kwargs = {"attn_implementation": "sdpa"}\n',
    1  # first occurrence (main loading loop)
)

# Patch 2: _apply_lora base_model loading (no extra_kwargs)
# Find: device_map="cpu",
# In the base_model from_pretrained block, add attn_implementation
content = content.replace(
    '                device_map="cpu",\n'
    '                trust_remote_code=self.trusted_models.get(self.settings.model),\n'
    '            )',
    '                device_map="cpu",\n'
    '                trust_remote_code=self.trusted_models.get(self.settings.model),\n'
    '                attn_implementation="sdpa",\n'
    '            )',
)

# Patch 3: reload() method - also has extra_kwargs
# Find the second extra_kwargs = {}
content = content.replace(
    '        extra_kwargs = {}\n',
    '        extra_kwargs = {"attn_implementation": "sdpa"}\n',
    1  # first remaining occurrence (reload method)
)

with open(model_py, 'w') as f:
    f.write(content)

print(f"Patched {model_py}")

# Verify
with open(model_py) as f:
    patched = f.read()

count = patched.count('attn_implementation')
print(f"Found {count} occurrences of attn_implementation")
if count < 3:
    print("WARNING: Expected at least 3 patches")
    sys.exit(1)
