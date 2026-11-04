#!/bin/bash
# Long-context Ornith dumps: chunked prefill, KV offload to CPU RAM, YaRN x4 (native context is 262144). One at a time on GPU 1.
# The HF attention mask is [chunk, T] and grows with context, so chunks shrink as T grows; on failure retry with half the chunk.
cd /ai/github/lsa-rocm || exit 1
export HIP_VISIBLE_DEVICES=1 PYTHONPATH=/ai/github/lsa-rocm/src
PY=/ai/environments/comfyui-rocm714/bin/python
for spec in "prose-coh-524288 524288 2048" "code-coh-524288 524288 2048" "code-coh-917504 917504 1024"; do
  set -- $spec; id=$1; L=$2; chunk=$3
  while [ ! -f results/dumps/ornith-yarn-$id/meta.json ] && [ $chunk -ge 256 ]; do
    echo "=== $id START chunk=$chunk $(date -Is)"
    rm -rf results/dumps/ornith-yarn-$id
    $PY -m lsa_rocm.dump --model /ai/models/llm/Ornith-1.5-9B --prompt-file experiments/data/texts/$id.txt \
       --max-tokens $L --n-queries 32 --prefill-chunk $chunk --kv-offload --gqa-sdpa --rope-yarn 4.0 --out results/dumps/ornith-yarn-$id --device cuda
    echo "=== $id EXIT $? chunk=$chunk $(date -Is)"
    [ -f results/dumps/ornith-yarn-$id/meta.json ] || chunk=$((chunk/2))
  done
done
echo ALL DONE
