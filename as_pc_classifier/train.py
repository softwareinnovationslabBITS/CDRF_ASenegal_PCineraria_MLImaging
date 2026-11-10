#!/usr/bin/env python3
"""
Training script for Acacia senegal vs. Prosopis cineraria Species Classifier.
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

parser = argparse.ArgumentParser(description="Train AS vs. PC Species Classifier (EfficientNetB0)")
parser.add_argument("--data_dir", type=str, default=None, help="Root dataset directory containing batches and excels")
parser.add_argument("--cpu", action="store_true", help="Force CPU training")
args = parser.parse_args()

if args.data_dir:
    os.environ["DATA_DIR"] = args.data_dir

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from core import train_backbone


if __name__ == "__main__":
    train_backbone("EfficientNetB0", SCRIPT_DIR)
