#!/usr/bin/env python3
"""Patch huggingface_hub to handle PEP 604 union types (X | Y) in strict dataclasses.

The kernels package (dependency of transformers) uses 'str | None' annotations
in classes that inherit from huggingface_hub's StrictDataclass, but the
type_validator in huggingface_hub doesn't handle types.UnionType (PEP 604).
"""

import sys
from pathlib import Path


def find_target():
    for candidate in sys.path:
        p = Path(candidate) / "huggingface_hub" / "dataclasses.py"
        if p.exists():
            return p
    return None


target = find_target()
if target is None:
    print("huggingface_hub/dataclasses.py not found, skipping patch")
    sys.exit(0)

code = target.read_text()

if "_PEP604_PATCHED" in code:
    print("Already patched, skipping")
    sys.exit(0)

lines = code.split("\n")
patched = False

for i, line in enumerate(lines):
    if "Unsupported type for field" in line and "raise TypeError" in line:
        indent = " " * (len(line) - len(line.lstrip()))
        new_lines = [
            f"{indent}# PEP 604 union type support (_PEP604_PATCHED)",
            f"{indent}import types as _types_mod",
            f"{indent}if isinstance(expected_type, _types_mod.UnionType):",
            f"{indent}    for _arg in expected_type.__args__:",
            f"{indent}        try:",
            f"{indent}            type_validator(name, value, _arg)",
            f"{indent}            return",
            f"{indent}        except TypeError:",
            f"{indent}            continue",
            line,
        ]
        lines[i : i + 1] = new_lines
        patched = True
        break

if not patched:
    print("Target pattern not found in type_validator, skipping patch")
    sys.exit(0)

target.write_text("\n".join(lines))
print(f"Patched {target} with PEP 604 union type support")
