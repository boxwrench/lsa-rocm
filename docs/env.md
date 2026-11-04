# Environment

Probed on 2026-11-03 by B0d (read-only; no GPU kernels, no downloads). Host `Nautilus`,
Ubuntu 24.04, kernel `7.0.0-28-generic`, Ryzen 7 9800X3D (16 threads), 186 GB RAM.

## GPUs

| ROCm / torch index | rocm-smi device | PCI bus | Name | gfx | VRAM total | Free at probe |
|---|---|---|---|---|---|---|
| 0 | GPU[0] | `0000:03:00.0` | AMD Radeon AI PRO R9700 | gfx1201 | 31.86 GiB (34,208,743,424 B) | ~30.3 GiB (desktop runs here, `boot_vga=1`) |
| 1 | GPU[1] | `0000:06:00.0` | AMD Radeon AI PRO R9700 | gfx1201 | 31.86 GiB (34,208,743,424 B) | ~31.8 GiB (idle) |
| 2 | GPU[2] | `0000:7e:00.0` | AMD Radeon Graphics (Raphael iGPU) | gfx1036 | 2 GiB carve-out (shares RAM) | ~2 GiB |

**There is no 7900 XT on this machine.** All of `rocm-smi --showproductname`, `lspci -nn`,
`rocminfo` and `vulkaninfo` show two R9700s and one Raphael iGPU; no Navi 31 / `gfx1100` device.

**Use device index 1 for GPU jobs** (idle card, desktop stays on 0). Set `HIP_VISIBLE_DEVICES=1`
(and `ROCR_VISIBLE_DEVICES=1` if you also want `rocm-smi`/HIP tooling restricted). Any run JSON
must record the device name, not just the index.

`rocm-smi` reports "free" only as total minus used; both `/sys/class/drm/card{1,2}/device/mem_info_vram_*`
and `rocm-smi --showmeminfo vram` agree. No process held `/dev/kfd` during the probe
(`rocm-smi --showpids`: "No KFD PIDs currently running").

### Vulkan memory (not visible to rocm-smi)

`vulkaninfo --summary` enumerates GPU0/GPU1 = "AMD Radeon AI PRO R9700 (RADV GFX1201)",
GPU2 = "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)", GPU3 = llvmpipe. Each R9700 exposes two
heaps:

- `memoryHeaps[0]`: 93.17 GiB, no flags -> host-visible system RAM (GTT)
- `memoryHeaps[1]`: 31.86 GiB, `DEVICE_LOCAL` -> real VRAM, matches rocm-smi

So a Vulkan allocator can report ~93 GiB more than rocm-smi; that is host memory (budget limited by
RAM), not VRAM. Size models against the 31.86 GiB device-local figure.

## ROCm

| item | value |
|---|---|
| install | `/opt/rocm` -> `/opt/rocm-7.2.1`; `/opt/rocm/.info/version` = `7.2.1` |
| userspace | `hsa-rocr 1.18.0.70201-81~24.04`, `rocm-smi` ROCM-SMI 4.0.0+e1a6bc5663 (SMI-LIB 7.8.0) |
| kernel driver | `amdgpu`; kernel `7.0.0-28-generic` |
| rocminfo | agents: 2x `gfx1201`, 1x `gfx1036`; ISA `gfx12-generic` |

## Python environments

`/ai/envs` is a symlink to `/ai/environments`. There is no project venv in this repo.

| env | python | torch | HIP | transformers | devices | notes |
|---|---|---|---|---|---|---|
| `/ai/environments/comfyui-rocm714` | 3.12.13 | 2.12.0+rocm7.14.0 | 7.14.60850 | 5.16.1 | 3 | **default for this project**: meets `granite-switch`'s `torch>=2.11`, `transformers>=5.16`; imports cleanly with no env vars; rocm-sdk 7.14.0 |
| `/ai/environments/kitchen-test` | 3.12.3 | 2.9.1+rocm7.2.1.gitff65f5bc | 7.2.53211 | 5.15.0 | 3 | closest to system `/opt/rocm` 7.2.1 and has pytest 9.1.1, scipy 1.18.0, scikit-learn 1.9.0, numpy 2.5.2, accelerate 1.14.0, datasets 5.0.1, huggingface_hub 1.27.0, safetensors 0.8.0, triton 3.5.1+rocm7.2.1; but **below granite-switch's torch/transformers floor** and `import torch` needs `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` |
| `/ai/environments/comfyui-qwen21` | 3.12.13 | 2.12.0+rocm7.14.0 | 7.14.60850 | 5.17.0 | 3 | also meets the granite-switch floor and has accelerate 1.15.0, scipy 1.18.1; prints a rocSHMEM "Could not open libnuma" warning |
| `/ai/environments/comfyui-rocm10` | 3.12.13 | 2.13.0+rocm10.1.0a20260823 | 7.16.26332 | 5.16.1 | 3 | pre-release rocm-sdk 10.1.0a20260823 |
| `/ai/environments/gguf-convert` | 3.12.3 | 2.13.0+cpu | None | 5.15.0 | 0 | CPU only |
| comfyui-h3, lingbot-world-v2, maestro | 3.12.x | torch present | — | — | — | `import torch` fails: `ImportError: libroctx64.so.4` unless `LD_LIBRARY_PATH=/opt/rocm/lib` |

Use (comfyui-rocm714; no env vars needed):

```sh
/ai/environments/comfyui-rocm714/bin/python -c "import torch, transformers; print(torch.__version__, torch.version.hip, transformers.__version__)"
# 2.12.0+rocm7.14.0 7.14.60850 5.16.1
```

kitchen-test needs `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64`:
`torch 2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, transformers `5.15.0`.

torch device order is the same as ROCm order (verified in comfyui-rocm714):

```
0 AMD Radeon AI PRO R9700   gfx1201  34208743424
1 AMD Radeon AI PRO R9700   gfx1201  34208743424
2 AMD Radeon Graphics       gfx1036  100043042816  (APU/GTT; never use)
```

### transformers support for the plan's models

Installed transformers versions: 5.15.0 (kitchen-test), 5.16.1 (comfyui-rocm714 and -rocm10),
5.17.0 (comfyui-qwen21). Checked the package trees in all four GPU envs (and the upstream
transformers trees for tags `v5.17.0` and `v5.22.1`): there is **no `granite_switch` module,
`GraniteSwitchForCausalLM` class, or AutoConfig entry**. The primary model declares
`"architectures": ["GraniteSwitchForCausalLM"]`, `"model_type": "granite_switch"`, so it needs the
separate `granite-switch` package.

Verified: with `granite-switch 0.1.1` (wheel already fetched by B0a at
`agents/scratch/B0a/gs011.whl`) on `PYTHONPATH`, comfyui-rocm714, comfyui-qwen21 and kitchen-test
all resolve `AutoConfig.for_model("granite_switch")` to `GraniteSwitchConfig`:

```sh
PYTHONPATH=/ai/github/lsa-rocm/agents/scratch/B0a/wheel011 \
  /ai/environments/comfyui-rocm714/bin/python -c \
  "import granite_switch.hf; from transformers import AutoConfig; \
   print(type(AutoConfig.for_model('granite_switch')).__name__)"
# GraniteSwitchConfig
```

Note `granite-switch 0.1.1` declares `torch>=2.11.0` and `transformers>=5.16,<=5.17.0`, so
kitchen-test (torch 2.9.1, transformers 5.15.0) is below the declared floor even though the import
works. comfyui-rocm714 (torch 2.12.0, transformers 5.16.1) and comfyui-qwen21 (5.17.0) are inside
the declared range; prefer them for granite runs. `gpt_oss`, `qwen3_5`, `granite`, `qwen3_next` are
all present in every env.

## Models on disk

| model | present | path | size |
|---|---|---|---|
| `ibm-granite/granite-switch-4.1-8b` | **no** | — | — |
| `Nanbeige/Nanbeige4.3-33B` | **no weights** | `models--Nanbeige--Nanbeige4.3-33B` has only `refs/main` + README blob | 36 KB |
| `ornith-ai/Ornith-1.5-9B` | **yes, complete** | `/ai/models/llm/Ornith-1.5-9B` | 19.31 GB (17.98 GiB) |
| `Qwen/Qwen3.5-0.8B` | **yes, complete** | `/ai/models/llm/Qwen3.5-0.8B`; also `Qwen3.5-0.8B-BF16.gguf` | 1.75 GB (1.63 GiB) |
| `openai/gpt-oss-20b` | **no** | — | — |

Both local models are `model_type: qwen3_5`, one `full_attention` layer per 4 (`full_attention_interval: 4`).
Ornith: 32 layers, 16 heads / 4 KV heads, head_dim 256. Qwen3.5-0.8B: 24 layers, 8 / 2, head_dim 256.
Load them by explicit path; the HF-cache entries for both contain only `refs/main`.

## bf16 sizes to download (Hub API, 2026-11-03)

| model | revision | bf16 bytes | GB | GiB |
|---|---|---|---|---|
| `ibm-granite/granite-switch-4.1-8b` | `fb4aa7877e13e5fe09ca328713dc1db72a3f93ce` | 17,554,232,752 (4 shards) | 17.55 | 16.35 |
| `Nanbeige/Nanbeige4.3-33B` | `df919e1e2cf077a22204c78b6c826bf3db18bc08` | 66,238,009,168 (14 shards) | 66.24 | 61.69 |
| `openai/gpt-oss-20b` (official) | `6cee5e81ee83917806bbde320786a8fb61efebee` | 13,761,316,904 (3 shards) — this is **MXFP4**, not bf16 | 13.76 | 12.82 |
| `gpt-oss-20b` true bf16 (`unsloth/gpt-oss-20b-BF16`) | `5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e` | 41,829,514,368 (9 shards) | 41.86 | 38.98 |

The official gpt-oss-20b checkpoint is native MXFP4 (u8 19.11e9 + bf16 1.80e9 params); genuine bf16
weights are a community conversion at 41.86 GB. B4c must say which form it used.

## Disk

```
$ df -h /ai
/dev/nvme1n1p1  1.8T  1.7T   71G  96% /ai
```

71 GB free. Granite bf16 fits; gpt-oss bf16 (39 GiB) fits only alone; Nanbeige bf16 (61.7 GiB)
leaves ~9 GB and cannot coexist with granite. Plan: free space or use a quant before B2/B4a.

## Tooling

`hf` CLI on PATH (user-local install); HF cache at `/ai/huggingface/hub` (137 repos).

## Second node: Strix Halo ("nimo")

Reachable over SSH on the local network. Probed read-only on 2026-11-03.

| item | value |
|---|---|
| CPU / memory | Ryzen AI MAX+ 395, 32 threads, 121 GiB unified RAM (87 GiB already in use at probe) |
| GPUs | RX 7900 XT (about 20 GiB VRAM), Radeon 8060S iGPU (4 GiB carve-out) |
| ROCm | 7.2.2 |
| Disk | 1.8 TB, 340 GB free |
| Tooling | `python3`, `git`. No `uv` or `hf`, no shared model store. |

Planned use: small-model sweeps on the 7900 XT, and later the CPU-memory-backed KV experiment, since the unified
memory is one pool. Needs its own Python environment and its own model copies. Not used by any experiment yet.

Hardware note: the machine in the table above ("Nautilus") had a 7900 XT until 2026-11-03, when it was swapped
for a second R9700. The 7900 XT is now on the Strix Halo node.
