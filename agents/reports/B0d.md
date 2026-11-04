# B0d — Machine probe

Status: complete. Read-only probe; no GPU kernels, no allocations, no downloads. Probe run
2026-11-03, ~17:00 PST, repo commit `4491fecd21e1be63c902dd851076c783bc347ce7`.

Task: ROCm and PyTorch versions, which GPU index is the R9700, free VRAM on each, transformers
version, which of the listed models are local under `/ai/huggingface` or `/ai/models`, and bf16
download sizes from the Hub API for the missing ones.

## Headline: there is no 7900 XT on this machine

The task asks which GPU index is the R9700 and which is the "7900 XT". The probe finds **two R9700s
and no 7900 XT**. (`docs/plan.md` only ever says "the R9700" and never names a 7900 XT.) The third
ROCm device is the Ryzen 9800X3D integrated GPU (Raphael, `gfx1036`).

```
$ rocm-smi --showproductname --showhw
GPU  NODE  DID     GUID   GFX VER  ...  VBIOS              BUS
0    1     0x7551  23334  gfx1201  ...  113-APM107573-101  0000:03:00.0
1    2     0x7551  2277   gfx1201  ...  113-APM107573-101  0000:06:00.0
2    3     0x13c0  38647  gfx1036  ...  102-RAPHAEL-008    0000:7E:00.0

GPU[0] Card Series: AMD Radeon AI PRO R9700   (gfx1201)
GPU[1] Card Series: AMD Radeon AI PRO R9700   (gfx1201)
GPU[2] Card Series: AMD Radeon Graphics        (gfx1036, Raphael iGPU)
```

`lspci` agrees: only two `1002:7551` (R9700) VGA devices at `03:00.0` and `06:00.0`, plus the
`1002:13c0` iGPU at `7e:00.0`. `vulkaninfo` agrees: GPU0/GPU1 = "AMD Radeon AI PRO R9700 (RADV
GFX1201)", GPU2 = "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)", GPU3 = llvmpipe. No Navi 31 /
`gfx1100` device appears anywhere in `rocminfo`, `rocm-smi`, `lspci` or Vulkan.

So the "R9700 index" is ambiguous: **GPU 0 and GPU 1 are both R9700**. See the recommendation below.

## ROCm

| item | value | how |
|---|---|---|
| ROCm install | `/opt/rocm` -> `/opt/rocm-7.2.1`, version file `7.2.1` | `cat /opt/rocm/.info/version` |
| ROCm userspace packages | `hsa-rocr 1.18.0.70201-81~24.04`, `rocm-smi` (ROCM-SMI 4.0.0+e1a6bc5663, SMI-LIB 7.8.0) | `dpkg -l`, `rocm-smi --version` |
| kernel driver | `amdgpu` module; `rocm-smi` prints "Driver version: 7.0.0-28-generic" (= kernel 7.0.0-28-generic) | `rocm-smi --showdriverversion` |
| torch HIP version | kernel-test: `7.2.53211-e1a6bc5663` (torch 2.9.1+rocm7.2.1); comfyui-*: `7.14.60850` (torch 2.12.0+rocm7.14.0); comfyui-rocm10: `7.16.26332` | `torch.version.hip` |
| `rocminfo` agents | 2x `gfx1201`, 1x `gfx1036`, ISA `gfx12-generic` present | `rocminfo` |

Note the ROCm userspace (7.2.1) and the newer venvs (rocm-sdk 7.14.0 / 10.1.0a) are different
stacks; see the env table.

## Python environments

`/ai/envs` is a symlink to `/ai/environments`. There is no project-local venv in `lsa-rocm` yet.

| env | python | torch | HIP | transformers | torch sees GPU? | notes |
|---|---|---|---|---|---|---|
| `/ai/environments/comfyui-rocm714` | 3.12.13 | 2.12.0+rocm7.14.0 | 7.14.60850 | **5.16.1** | yes, 3 devices | **default: meets granite-switch's torch>=2.11 / transformers>=5.16**; imports cleanly, no env vars needed |
| `/ai/environments/kitchen-test` | 3.12.3 | 2.9.1+rocm7.2.1.gitff65f5bc | 7.2.53211 | 5.15.0 | yes, 3 devices | closest to system ROCm 7.2.1, has pytest/scipy/sklearn/triton, but **below granite-switch floor** and **torch import fails without `LD_LIBRARY_PATH=/opt/rocm/lib`** |
| `/ai/environments/comfyui-qwen21` | 3.12.13 | 2.12.0+rocm7.14.0 | 7.14.60850 | 5.17.0 | yes, 3 devices | also meets the granite-switch floor, has accelerate 1.15.0 + scipy; prints a rocSHMEM/libnuma warning |
| `/ai/environments/comfyui-rocm10` | 3.12.13 | 2.13.0+rocm10.1.0a20260823 | 7.16.26332 | 5.16.1 | yes, 3 devices | pre-release rocm-sdk 10.1.0a20260823 |
| `/ai/environments/gguf-convert` | 3.12.3 | 2.13.0+cpu | None | 5.15.0 | no | CPU only |
| comfyui-h3, lingbot-world-v2, maestro | 3.12.x | — | — | — | import fails | `ImportError: libroctx64.so.4` unless `LD_LIBRARY_PATH=/opt/rocm/lib` |

Verified device names from `torch.cuda` (env comfyui-rocm714):

```
0 AMD Radeon AI PRO R9700        gfx1201  total 34208743424  pci 3
1 AMD Radeon AI PRO R9700        gfx1201  total 34208743424  pci 6
2 AMD Radeon Graphics            gfx1036  total 100043042816  pci 126   <- iGPU, shares system RAM
```

ROCm index == torch index == rocm-smi index == `rocminfo` order (0,1 = R9700; 2 = iGPU). Device 2
reports 100 GB "memory" because it is the APU (GTT); it must be excluded from any GPU work.

**Recommended env for this project: `/ai/environments/comfyui-rocm714/bin/python`.** It is the
lowest-friction ROCm env (no `LD_LIBRARY_PATH` needed), and at torch 2.12.0 / transformers 5.16.1 it
is the oldest env that satisfies `granite-switch 0.1.1`'s declared `torch>=2.11`,
`transformers>=5.16,<=5.17.0` — which is required for the primary model. It does not have
pytest/scipy/sklearn preinstalled, so B1a may need `pip install pytest` into a project venv copied
from it (or use kitchen-test for pure-CPU metric tests). kitchen-test remains the closest match to
system ROCm 7.2.1 but is below the granite-switch floor.

## Free VRAM

`rocm-smi --showmeminfo vram` at 17:00 PST:

```
GPU[0] Total 34208743424 (31.86 GiB)  Used 1724850176 (1.61 GiB)  Free 32483893248 (30.25 GiB)
GPU[1] Total 34208743424 (31.86 GiB)  Used   59912192 (0.06 GiB)  Free 34148831232 (31.80 GiB)
GPU[2] Total  2147483648 ( 2.00 GiB)  Used   16658432 (0.02 GiB)  Free  2130825216 ( 1.98 GiB)
```

Re-sampled 2 min later: GPU0 used 1,688,932,352 B (free 30.29 GiB), GPU1 used 59,912,192 B (free
31.80 GiB) — so GPU0's usage moves with the desktop; treat ~30 GiB as its working figure.

GPU 0 (PCI `03:00.0`) is `boot_vga=1` and holds the desktop session: ~1.6 GiB resident. GPU 1 (PCI
`06:00.0`) is idle. "Free" is total minus used, so ~30.25 GiB / ~31.80 GiB usable respectively
(rocm-smi has no separate free field; `/sys/class/drm/card{1,2}/device/mem_info_vram_*` matches).

Recommendation: run GPU jobs on **device index 1** (the idle R9700) so the desktop on device 0 does
not compete for VRAM or perturb timings, and set `HIP_VISIBLE_DEVICES=1`. For B3 timing, record
both and prefer index 1.

## Vulkan VRAM (not visible to rocm-smi)

`vulkaninfo --summary` shows 4 devices: R9700, R9700, Raphael iGPU, llvmpipe. For each R9700 the
Vulkan memory heaps are:

```
memoryHeaps[0]: size 100043042816 (93.17 GiB)  flags: None                  <- host-visible / GTT (system RAM)
memoryHeaps[1]: size  34208743424 (31.86 GiB)  flags: DEVICE_LOCAL           <- real VRAM
```

So Vulkan's device-local heap (31.86 GiB) matches rocm-smi, and the extra 93 GiB heap is host-visible
system RAM (MAP_HOST / GTT), which rocm-smi never reports. That is the "Vulkan VRAM invisible to
rocm-smi" caveat: a Vulkan/llama.cpp-style allocator can appear to have ~93 GB more memory than
rocm-smi shows, but it is host memory, not VRAM. Any model-size plan must use the 31.86 GiB figure.

## transformers

| env | transformers |
|---|---|
| kitchen-test | 5.15.0 |
| comfyui-rocm714 | 5.16.1 |
| comfyui-rocm10 | 5.16.1 |
| comfyui-qwen21 | 5.17.0 |
| gguf-convert | 5.15.0 (CPU) |

Checked (grep of `transformers/models` + registry) in all four GPU envs: there is **no
`granite_switch` / `GraniteSwitch` modeling file or AutoConfig entry** (granite dirs present:
`granite`, `granite_swa`, `granitemoe*`, `granite4_vision`, `granite_speech*`). The primary model
`ibm-granite/granite-switch-4.1-8b` declares `"architectures": ["GraniteSwitchForCausalLM"]`,
`"model_type": "granite_switch"`, so it needs the separate `granite-switch` package (or vendored
`modeling_granite_switch.py`, which B0a fetched into `agents/scratch/B0a/`).

Verified fix: `granite-switch 0.1.1` (wheel already in `agents/scratch/B0a/gs011.whl`) registers the
model. With it on `PYTHONPATH`:

```
$ PYTHONPATH=.../wheel011 /ai/environments/comfyui-rocm714/bin/python -c \
    "import granite_switch.hf; from transformers import AutoConfig; \
     print(type(AutoConfig.for_model('granite_switch')).__name__)"
AutoConfig granite_switch -> GraniteSwitchConfig
```

Same result in kitchen-test (with `LD_LIBRARY_PATH`) and comfyui-qwen21. Without importing
`granite_switch.hf` first, `AutoConfig.for_model("granite_switch")` raises
`ValueError: Unrecognized model identifier`.

`gpt_oss`, `qwen3_5`, `granite`, `qwen3_next` modules are present in all four GPU envs.

## Models on disk

Searched `/ai/huggingface/hub`, `/ai/models` (and `/ai/models/llm`) for each name.

| model (plan role) | present? | where | size on disk |
|---|---|---|---|
| `ibm-granite/granite-switch-4.1-8b` (primary) | **NO** — no `models--ibm-granite--granite-switch-4.1-8b` dir, no files anywhere under `/ai/models` or `/ai/huggingface` | — | — |
| `Nanbeige/Nanbeige4.3-33B` (scale) | **NO weights** | `/ai/huggingface/hub/models--Nanbeige--Nanbeige4.3-33B/` has only `refs/main` + one 8.8 KB blob (README). No `snapshots/` content, no safetensors. | 36 KB |
| `ornith-ai/Ornith-1.5-9B` (hybrid) | **YES** | `/ai/models/llm/Ornith-1.5-9B` (complete: 4 shards, 775 tensors in index, all shards present) | 19.31 GB (17.98 GiB) |
| `Qwen/Qwen3.5-0.8B` (harness) | **YES** | `/ai/models/llm/Qwen3.5-0.8B` (1 shard, 488 tensors, complete). Also `/ai/models/llm/Qwen3.5-0.8B-BF16.gguf` (1.56 GB) | 1.75 GB (1.63 GiB) |
| `openai/gpt-oss-20b` (control) | **NO** | only `models--openai--gsm8k` / `--openai_humaneval` datasets; no gpt-oss weights | — |

Also note `models--Qwen--Qwen3.5-0.8B` and `models--ornith-ai--Ornith-1.5-9B` exist in the HF cache
but contain only `refs/main` (+ README blob for Ornith); the usable copies are the `/ai/models/llm`
directories. Neither is a HF-format cache snapshot, so load with an explicit local path.

Integrity check (read-only): parsed every local safetensors header. Ornith shards hold
398+148+156+73 = 775 tensors, matching its index; Qwen3.5-0.8B holds 488, matching its index. All
headers parse and every shard named in the index exists at the size implied by it.

Both local models are `model_type: qwen3_5` (`Qwen3_5ForConditionalGeneration`), 1 full_attention
layer in every 4 (`full_attention_interval: 4`), head_dim 256 — the plan's "1 in 4 layers full" is
confirmed for Ornith by its `config.json`/`text_config.layer_types` (32 layers, 16 heads, 4 KV heads)
and for Qwen3.5-0.8B (24 layers, 8 heads, 2 KV heads).

## bf16 download sizes for the missing models (Hub API, not downloaded)

From `GET https://huggingface.co/api/models/<id>/tree/main?recursive=true` (sizes in bytes), and
`GET /api/models/<id>` for `safetensors.parameters`.

| model | repo revision | bf16 file bytes | GB (1e9) | GiB (2^30) | notes |
|---|---|---|---|---|---|
| `ibm-granite/granite-switch-4.1-8b` | `fb4aa7877e13e5fe09ca328713dc1db72a3f93ce` | 17,554,232,752 (4 shards) | **17.55 GB** | 16.35 GiB | repo total 17.56 GB; BF16 params 8,776,400,896 |
| `Nanbeige/Nanbeige4.3-33B` | `df919e1e2cf077a22204c78b6c826bf3db18bc08` | 66,238,009,168 (14 shards) | **66.24 GB** | 61.69 GiB | repo total 66.26 GB; config `torch_dtype: bfloat16` |
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | official repo is **MXFP4**: 13,761,316,904 (3 shards) | 13.76 GB | 12.82 GiB | NOT bf16 (u8 19.11e9 + bf16 1.80e9 params) |
| `openai/gpt-oss-20b` as true bf16 | mirror `unsloth/gpt-oss-20b-BF16` @ `5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e` | 41,829,514,368 (9 shards), repo total 41,857,495,732 | **41.86 GB** | 38.98 GiB | 20,914,757,184 params x 2 bytes |

So: if "gpt-oss-20b bf16" means genuine bf16 weights, budget **41.86 GB (38.98 GiB)**; the
official checkpoint download is only 13.76 GB because it is MXFP4-quantized. (The plan only allows
gpt-oss-20b as a control, and B4c would have to state which form it uses.)

## Disk space problem (blocking for downloads)

```
$ df -h /ai
/dev/nvme1n1p1  1.8T  1.7T   71G  96% /ai
```

71 GB free. Granite bf16 (16.4 GiB) fits; Nanbeige bf16 (61.7 GiB) technically fits but leaves ~9 GB;
gpt-oss bf16 (39 GiB) fits only alone. Downloading Nanbeige + granite together (~78 GiB) does not
fit. Needs a cleanup or a different quant (B0b reports which quants exist). Flagged, not acted on.

## Other environment facts

- Host `Nautilus`, Ubuntu 24.04, kernel `7.0.0-28-generic`, 16 threads (Ryzen 7 9800X3D), 186 GB RAM.
- Both R9700s at PCIe 4.0 x16 (`current_link_speed 32.0 GT/s`, `current_link_width 16`).
- `/dev/kfd` and `/dev/dri/{card1,card2,card3,renderD128-130}` exist; no process held `/dev/kfd`
  during the probe (`rocm-smi --showpids`: "No KFD PIDs currently running").
- No GPU work was run. Only `rocm-smi`, `rocminfo`, `vulkaninfo`, `lspci`, `df`, `ls`, and torch
  *device-name enumeration* (no allocation, no kernel).

## Open questions

1. The task expects a 7900 XT; the machine has 2x R9700. The plan's rule "GPU jobs run on the R9700
   only" is satisfiable, but "the R9700" implies a single device index. Recommend: GPU index 1
   (idle), `HIP_VISIBLE_DEVICES=1`, and a one-line clarification in the plan. I did not edit
   `docs/plan.md` (B0d is read-only for it).
2. Which env is canonical? I recommend comfyui-rocm714 because it meets the granite-switch floor
   and imports without `LD_LIBRARY_PATH`; kitchen-test still needs the `LD_LIBRARY_PATH` note.
   This is a judgement call and should be confirmed before B1.
3. `granite-switch` must be installed (or vendored); it is not in any env and is not a transformers
   built-in. Verified that 0.1.1 registers the model when imported.
4. Nanbeige4.3-33B and gpt-oss-20b cannot be downloaded without freeing disk or choosing a quant
   (only 71 GB free). Nothing was downloaded.
5. `nanbeige` HF cache entry is a stub (refs + README only), so a `from_pretrained("Nanbeige/...")`
   would silently re-download 66 GB. Use an explicit path or check first.
