"""Stable package and repository locations, independent of module depth."""
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parents[1]
