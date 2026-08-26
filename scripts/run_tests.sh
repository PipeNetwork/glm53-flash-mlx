#!/bin/sh
# Tiny-scale parity against transformers (needs torch + transformers, see requirements.txt).
cd "$(dirname "$0")/.." && exec .venv/bin/python tests/test_parity.py
