"""Shared importlib loader for voice-plugin modules under test.

The voice plugin's submodules are loaded flat from their file paths (no
package context), mirroring how tests/gateway/_plugin_adapter_loader.py
loads adapter.py — so tests exercise the same flat-import reality as the
test loader. Modules are cached in sys.modules as ``voice_plugin_<name>``
so repeated loads (across test files in one process) return the same
module object.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_voice_module(name: str):
    mod_name = f"voice_plugin_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _REPO_ROOT / "plugins" / "platforms" / "voice" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod
