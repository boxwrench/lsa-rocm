# B1c — Q/K/V dumper

Task: implement `src/lsa_rocm/dump.py`; dump full-attention Q/K/V for a local HF model
with one forward pass; self-check one layer; run on Qwen3.5-0.8B (~2048 tok, CPU) and
Ornith-1.5-9B (~4096 tok, GPU index 1).

## Inherited state and what was fixed

- `src/lsa_rocm/dump.py` already existed from the previous attempt (367 lines) and its
  forward-hook design was correct. It had one bug in the self-check: the attention
  matmul used the wrong dimensions. The file now has
  `logits = (q_last.unsqueeze(1) @ k_rep.transpose(1, 2)).squeeze(1) * rec["scaling"]`
  and `out_recomputed = (probs.unsqueeze(1) @ v_rep).squeeze(1)`.
  (The first crash was `Expected size ... [2048, 8]`; after the first fix a second
  crash showed `own (8, 256) vs (8, 8, 256)` from `probs @ v_rep`, fixed as above.)
- `agents/scratch/prompt_long.txt` (170735 B, repeated local paragraph) was used as the
  prompt; `/ai/datasets` is empty.
- `results/dumps/qwen3.5-0.8b-2048/` existed but was EMPTY (previous run crashed before
  saving). No Ornith dump existed.

## Design (as implemented)

- Full-attention layers from `config.text_config.layer_types`; falls back to all layers
  when absent. Attention modules are found by `layer_idx` and the presence of
  q/k/v/o projections. The vision tower's `Qwen3_5VisionAttention` has no `layer_idx`
  and is never selected.
- Capture is done by wrapping `ALL_ATTENTION_FUNCTIONS["eager"]` with a call to the
  model's own `eager_attention_forward`, which receives post-RoPE Q/K/V and the module's
  own `scaling`; the returned `out` is the module's own attention output before the
  Qwen3.5 output gate.
- Pre-RoPE keys are captured by wrapping the modeling module's `apply_rotary_pos_emb`
  while a forward pre-hook marks the current layer; `rotary_dim = cos.shape[-1]`.
- Per layer: `keys_post_rope [Hkv,T,D]`, `keys_pre_rope [Hkv,T,D]`, `values [Hkv,T,D]`,
  `queries_post_rope [Hq,Q,D]`, `positions [Q]`, `scaling`, head/group counts.
- Self-check recomputes the last query's attention from the dumped tensors with the
  captured scaling (softmax in float32, GQA repeat_interleave) and compares to the
  module's own attention output; aborts if cosine <= 0.999.

## Environment

- `/ai/environments/comfyui-rocm714/bin/python`: transformers 5.16.1,
  torch 2.12.0+rocm7.14.0, ROCm 7.14.60850.
- The CPU env `agents/scratch/venv-lsa-min` has torch but **no transformers**, so both
  runs used the ROCm python (CPU tensors for the Qwen run).
- `rocm-smi` was NOT idle at start: PID 1741791 (an unrelated `moss-test/ab.py` from
  another project) held 8.9 GB then up to 17 GB VRAM on physical GPU 1. It exited at
  18:51:17 (its `timeout 900`), after which GPU 1 showed 59.9 MB used and the Ornith
  run started. Devices 0 and 2 were never used.

## Commands and output

### Qwen3.5-0.8B, 2048 tokens, CPU

```
cd /ai/github/lsa-rocm && time PYTHONPATH=/ai/github/lsa-rocm/src \
  /ai/environments/comfyui-rocm714/bin/python -m lsa_rocm.dump \
  --model /ai/models/llm/Qwen3.5-0.8B --prompt-file agents/scratch/prompt_long.txt \
  --max-tokens 2048 --n-queries 8 --out results/dumps/qwen3.5-0.8b-2048 --device cpu
```

```
SELF-CHECK layer 3: cosine(last-query attention output) = 0.99999994 (threshold 0.999)
WROTE results/dumps/qwen3.5-0.8b-2048/meta.json (72.39 MiB of tensors across 6 files)
{"full_attention_layer_indices": [3, 7, 11, 15, 19, 23], "self_check_cosine": 0.9999999403953552, "dump_mib": 72.392}
EXIT=0, real 58.6s
```

### Ornith-1.5-9B, 4096 tokens, GPU index 1 (HIP_VISIBLE_DEVICES=1)

```
cd /ai/github/lsa-rocm && time HIP_VISIBLE_DEVICES=1 PYTHONPATH=/ai/github/lsa-rocm/src \
  /ai/environments/comfyui-rocm714/bin/python -m lsa_rocm.dump \
  --model /ai/models/llm/Ornith-1.5-9B --prompt-file agents/scratch/prompt_long.txt \
  --max-tokens 4096 --n-queries 8 --out results/dumps/ornith-1.5-9b-4096 --device cuda
```

```
SELF-CHECK layer 3: cosine(last-query attention output) = 0.99999738 (threshold 0.999)
WROTE results/dumps/ornith-1.5-9b-4096/meta.json (192.52 MiB of tensors across 8 files)
{"full_attention_layer_indices": [3, 7, 11, 15, 19, 23, 27, 31], "self_check_cosine": 0.9999973773956299, "dump_mib": 192.522}
EXIT=0, real 6m54s
```

## Verification of the dumps (read back from disk)

Qwen `layer_03.pt`: `queries_post_rope (8,8,256) f32`, `keys_post_rope (2,2048,256) f32`,
`values (2,2048,256) f32`, `keys_pre_rope (2,2048,256) f32`, `positions (8,)`,
`scaling 0.0625`, groups 4.

Ornith `layer_03.pt`: `queries_post_rope (16,8,256) bf16`, `keys_post_rope (4,4096,256) bf16`,
`values (4,4096,256) bf16`, `keys_pre_rope (4,4096,256) bf16`, `positions (8,)`,
`scaling 0.0625`, groups 4.

meta.json (both): `model_path`, `model_revision = null` (local directory has no HF
`_commit_hash`), `layer_types`, `full_attention_layer_indices`, `layer_index_map`,
`attention_scaling_per_layer` (0.0625 = 256^-0.5 for every full layer, read from the
module call), `gqa_group_size`, `rope` (theta 1e7, partial_rotary_factor 0.25,
mrope_section [11,11,10]), `rotary_dim_per_layer` = 64 for every layer, `dtype`
(float32 CPU / bfloat16 GPU), `device`, `device_name` ("AMD Radeon AI PRO R9700" on GPU),
`torch_version`, `rocm_version`, `seed` 0, `self_check`.

## Dump sizes

- `results/dumps/qwen3.5-0.8b-2048/`: 73 MB on disk (72.39 MiB across 6 layer files +
  meta.json).
- `results/dumps/ornith-1.5-9b-4096/`: 193 MB on disk (192.52 MiB across 8 layer files +
  meta.json).

## Acceptance

- Both runs completed (EXIT=0), shown above.
- Self-check cosine printed for each: 0.99999994 (Qwen3.5-0.8B) and 0.99999738
  (Ornith-1.5-9B), both above 0.999.
- meta.json lists full-attention indices: `[3,7,11,15,19,23]` = 6 of 24 for Qwen3.5-0.8B
  and `[3,7,11,15,19,23,27,31]` = 8 of 32 for Ornith-1.5-9B.

## Open points

- `model_revision` is null because the models are local directories; the path and a git
  commit of this repo are recorded instead.
- The GPU was not idle when the task started; the unrelated job was waited out (it
  exited at 18:51:17) before the Ornith run, which then saw an idle device.
- Only the first full-attention layer is self-checked, as specified.

STATUS: DONE
