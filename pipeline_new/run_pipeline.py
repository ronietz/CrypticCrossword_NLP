#!/usr/bin/env python3
"""Launcher. Runs the same whether you invoke it by path or as a module.

    python code/pipeline/run_pipeline.py --clue "..."     # by path, from the repo root
    python -m pipeline.run_pipeline --clue "..."          # as a module, from code/

All the logic is in cli.py; this file only fixes up sys.path for the by-path
case. Note the package is `pipeline`, rooted at `code/`, NOT `code.pipeline` --
`code` is a Python standard-library module name, and shadowing it breaks any
dependency that imports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.cli import main
else:
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
