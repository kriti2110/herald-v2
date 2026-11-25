#!/bin/bash
cd "$(dirname "$0")/.."
t=$1
python3 scripts/07_final.py --mode cv --task $t --threads 5
python3 scripts/07_final.py --mode transfer --task $t --threads 5
echo FINAL_DONE_$t
