"""
Blackwell (sm_120) compatibility patch for Heretic.

bitsandbytes has no CUDA 13.1 binary and heretic imports it unconditionally.
Stub it out so `import bitsandbytes` doesn't crash.
Includes bnb.nn submodule stub for peft compatibility.
"""

import importlib.machinery
import sys
import types

if "bitsandbytes" not in sys.modules:
    bnb_stub = types.ModuleType("bitsandbytes")
    bnb_stub.__version__ = "0.0.0-stub"
    bnb_stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)

    class _FakeBitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes is not available (no CUDA 13.1 binary)."
            )

    bnb_stub.BitsAndBytesConfig = _FakeBitsAndBytesConfig

    # Stub bnb.nn submodule - peft checks for bnb.nn.Linear4bit / Linear8bitLt
    nn_stub = types.ModuleType("bitsandbytes.nn")
    nn_stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.nn", None)
    bnb_stub.nn = nn_stub
    sys.modules["bitsandbytes.nn"] = nn_stub

    # Stub bnb.functional - some code checks for bnb.functional
    func_stub = types.ModuleType("bitsandbytes.functional")
    func_stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.functional", None)
    bnb_stub.functional = func_stub
    sys.modules["bitsandbytes.functional"] = func_stub

    sys.modules["bitsandbytes"] = bnb_stub
