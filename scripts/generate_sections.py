#!/usr/bin/env python3

"""Repository-root entry point for virtual section generation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sections.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
