#!/usr/bin/env python3
"""CLI entry point for scheduled or manual model training."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Suppress MLFlow Git warning before imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Scripts.train import main as train_main


def build_parser():
    parser = argparse.ArgumentParser(description='Run the churn training pipeline.')
    parser.add_argument(
        '--reason',
        default='manual',
        help='Human-readable reason for the training run, used for workflow logs.',
    )
    parser.add_argument(
        '--profile',
        choices=['quick', 'balanced', 'full'],
        default=os.getenv('TRAINING_PROFILE', 'quick'),
        help='Training search profile. quick is fastest, full is the most exhaustive.',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"Starting model training at {datetime.now()} (reason: {args.reason}, profile: {args.profile})")
    train_main(profile=args.profile)
    print(f"Training completed at {datetime.now()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
