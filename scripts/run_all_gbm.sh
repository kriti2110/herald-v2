#!/bin/bash
cd "$(dirname "$0")/.."
for task in forecast detect; do
  for v in full no_care physio_only; do
    python3 scripts/03_gbm.py --mode cv --task $task --variant $v
  done
  python3 scripts/03_gbm.py --mode transfer --task $task --variant full
  python3 scripts/03_gbm.py --mode transfer --task $task --variant physio_only
done
echo ALL_GBM_DONE
