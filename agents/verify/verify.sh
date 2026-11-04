#!/bin/bash
case "$1" in
 B1e) exec /ai/environments/comfyui-rocm714/bin/python -I /ai/github/lsa-rocm/agents/verify/B1e.py;;
 B1f) exec /ai/environments/comfyui-rocm714/bin/python -I /ai/github/lsa-rocm/agents/verify/B1f.py;;
 B2_0) exec /ai/environments/comfyui-rocm714/bin/python -I /ai/github/lsa-rocm/agents/verify/B2_0.py;;
esac
