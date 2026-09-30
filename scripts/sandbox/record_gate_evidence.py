#!/usr/bin/env python3
"""Called by the outer shell only after every qualification gate succeeds."""
import argparse
from pathlib import Path
from env_factory.evidence.gate_cache import GATES, save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    save(args.root, args.project, {name: [True, 'Passed by outer build workflow for bound inputs'] for name in GATES})
