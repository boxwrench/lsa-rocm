#!/bin/bash
# usage: run_resilient.sh OUT.json N_PROMPTS python-args...   Re-runs run_matrix.py with --resume until OUT has N_PROMPTS results (max 8 tries).
OUT=$1; N=$2; shift 2
for try in 1 2 3 4 5 6 7 8; do
  HIP_VISIBLE_DEVICES=1 /ai/environments/comfyui-rocm714/bin/python /ai/github/lsa-rocm/experiments/005-needle/run_matrix.py --resume --out "$OUT" "$@"
  have=$(python3 -c "import json,os;print(len(json.load(open('$OUT'))['results']) if os.path.exists('$OUT') else 0)")
  echo "TRY $try: $have/$N prompts"; [ "$have" -ge "$N" ] && break; sleep 20
done
