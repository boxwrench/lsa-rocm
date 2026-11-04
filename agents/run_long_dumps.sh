#!/bin/bash
# Long-context Ornith dumps: chunked prefill, KV offload to CPU RAM, YaRN x4 (native context is 262144). One at a time on GPU 1.
cd /ai/github/lsa-rocm || exit 1
while pgrep -f "agents/scratch/b25a_run2.sh" >/dev/null; do sleep 15; done
export HIP_VISIBLE_DEVICES=1 PYTHONPATH=/ai/github/lsa-rocm/src
PY=/ai/environments/comfyui-rocm714/bin/python
for spec in "prose-coh-524288 524288" "code-coh-524288 524288" "code-coh-917504 917504"; do
  set -- $spec; id=$1; L=$2
  [ -f results/dumps/ornith-yarn-$id/meta.json ] && { echo "skip $id (done)"; continue; }
  echo "=== $id START $(date -Is)"
  $PY -m lsa_rocm.dump --model /ai/models/llm/Ornith-1.5-9B --prompt-file experiments/data/texts/$id.txt \
     --max-tokens $L --n-queries 32 --prefill-chunk 8192 --kv-offload --rope-yarn 4.0 --out results/dumps/ornith-yarn-$id --device cuda
  echo "=== $id EXIT $? $(date -Is)"
done
echo ALL DONE
