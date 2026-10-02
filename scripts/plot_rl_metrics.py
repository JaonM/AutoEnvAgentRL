#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in src/rl."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl.plot_metrics import main

if __name__ == "__main__":
    main()
