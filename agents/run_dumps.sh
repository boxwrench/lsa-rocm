#!/bin/bash
# Sequential Ornith dumps for the coherent prompts. One process at a time on GPU 1.
cd /ai/github/lsa-rocm || exit 1
export HIP_VISIBLE_DEVICES=1 PYTHONPATH=/ai/github/lsa-rocm/src
PY=/ai/environments/comfyui-rocm714/bin/python
for spec in "code-coh-32768 32768" "tech-coh-32768 32768" "code-coh-131072 131072" "tech-coh-131072 131072"; do
  set -- $spec; id=$1; L=$2
  echo "=== $id START $(date -Is)"
  $PY -m lsa_rocm.dump --model /ai/models/llm/Ornith-1.5-9B --prompt-file experiments/data/texts/$id.txt \
     --max-tokens $L --n-queries 32 --prefill-chunk 8192 --out results/dumps/ornith-$id --device cuda
  echo "=== $id EXIT $? $(date -Is)"
done
echo ALL DONE
