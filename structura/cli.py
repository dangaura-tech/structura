"""Console entry point wrapping ``tools/reconstruct.py`` (available with an editable install)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def reconstruct() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "reconstruct.py"
    if not path.exists():
        raise SystemExit("tools/reconstruct.py not found - install with `pip install -e .`")
    spec = importlib.util.spec_from_file_location("structura_reconstruct", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()
