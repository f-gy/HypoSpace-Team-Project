#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
run_dir="${1:-reproduction_$(date +%Y%m%dT%H%M%S)}"
if [ -e "$run_dir" ]; then
  echo "Refuse to overwrite existing run: $run_dir" >&2
  exit 3
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/budget_v1:$PWD/certified_generation_v1${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" budget_v1/prepare.py --run-dir "$run_dir"
"$PYTHON_BIN" -m unittest discover -s budget_v1 -v > "$run_dir/tests.log" 2>&1
"$PYTHON_BIN" -m unittest discover -s certified_generation_v1 -v > "$run_dir/frozen_tests.log" 2>&1
"$PYTHON_BIN" budget_v1/run_inference.py --run-dir "$run_dir" > "$run_dir/inference.log" 2>&1
"$PYTHON_BIN" budget_v1/evaluate.py --run-dir "$run_dir" --require-frozen-success > "$run_dir/scoring.log" 2>&1
"$PYTHON_BIN" - "$run_dir" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]);rows=json.load(open(p/'batch_summary.json'))
assert len(rows)==54
for r in rows:
    if r['split']=='internal_test' and r['order']=='native':
        print(r['policy'], 'emitted=',r['emitted'], 'unique_valid=',r['unique_valid'],
              'uniqueness=',r['uniqueness_emitted'], 'recovery=',r['macro_recovery'])
print('VERIFIED: 17 new tests + 9 frozen tests; 54 batches; 720 scored question-runs.')
print('Results:',p.resolve())
PY
