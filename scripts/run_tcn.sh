#!/bin/bash
cd "$(dirname "$0")/.."
python3 scripts/05_tcn.py --mode cv
python3 scripts/05_tcn.py --mode transfer
echo ALL_TCN_DONE
