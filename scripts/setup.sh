#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
python3 -m venv .venv
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/loadcell-pip-cache}" .venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
MPLBACKEND=Agg .venv/bin/python scripts/check_environment.py
