#!/bin/bash
PY=/ai/environments/comfyui-rocm714/bin/python
case "$1" in
 B1e|B1f|B1g|B2_0|B2_1|B2_2) exec $PY -I /ai/github/lsa-rocm/agents/verify/$1.py;;
esac
