#!/bin/bash
# usage: run_agent2.sh ID   (ID in B1e B1f B2_0). Supervisor-owned gate: verify.sh decides PASS/FAIL, not the agent.
cd /ai/github/lsa-rocm
id=$1; R=/ai/github/lsa-rocm
export OPENCODE_CONFIG_CONTENT='{"permission":{"external_directory":{"/ai/huggingface/**":"allow","/ai/models/**":"allow","/ai/datasets/**":"allow","/ai/github/**":"allow","/ai/environments/**":"allow","/opt/**":"allow"}}}'
BASE="$(cat agents/prompts/COMMON2.txt agents/prompts/$id.txt)"
st(){ echo "$(date +%H:%M) $id: $*" >> agents/reports/wrapper.log; }
gpu_wait(){ for i in $(seq 1 60); do u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\].*Used/{print $NF}'); [ -n "$u" ] && [ "$u" -lt 1000000000 ] && return 0; sleep 20; done; return 1; }
vcheck(){ sha256sum -c --quiet agents/verify/SHA256SUMS >/dev/null 2>&1 || { st "verify scripts modified, abort"; echo TAMPER > agents/reports/$id.status; exit 1; }
          bash agents/verify/verify.sh $id > agents/reports/$id.verify.txt 2>&1; }
attempt(){ # model tag extra
  timeout 1380 opencode run -m "$1" "$BASE
$3" > agents/reports/$id.$2.log 2>&1; }
[ "$id" = "B1e" ] && { gpu_wait || { st "GPU 1 not idle after 20 min"; echo BLOCKED > agents/reports/$id.status; exit 1; }; }
DS=merge-gateway/deepseek/deepseek-v4.1-flash-1017; LC=longcat/LongCat-2.5-Preview
n=0
for spec in "$DS ds1" "$DS ds2" "$LC lc1"; do
  set -- $spec; model=$1; tag=$2; n=$((n+1))
  # skip the DeepSeek retry unless the previous failure was a network error
  if [ "$tag" = ds2 ] && ! grep -qE "EPROTO|ECONNRESET|fetch failed" agents/reports/$id.ds1.log 2>/dev/null; then continue; fi
  [ "$tag" = ds2 ] && sleep 60
  [ "$id" = "B1e" ] && { gpu_wait || { st "GPU busy"; echo BLOCKED > agents/reports/$id.status; exit 1; }; }
  extra=""
  if [ -f agents/reports/$id.verify.txt ]; then extra="CONTINUATION: an earlier attempt left files and a partial report. The last acceptance run printed:
$(tail -30 agents/reports/$id.verify.txt)
Read agents/reports/$id.md, reuse what is correct, fix what failed, and finish."; fi
  st "attempt $tag ($model)"
  attempt "$model" $tag "$extra"
  vcheck
  if [ $? -eq 0 ]; then st "PASS after $tag"; echo "PASS $tag" > agents/reports/$id.status; exit 0; fi
  st "verify FAILED after $tag: $(tail -1 agents/reports/$id.verify.txt | cut -c1-200)"
done
echo FAIL > agents/reports/$id.status; st "FAIL after all attempts"
