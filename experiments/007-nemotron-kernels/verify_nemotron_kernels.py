import os
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # enables efficient/flash SDPA on gfx1151 (attention layers only)
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/home/keith/lsa/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from lsa_rocm.kernels import nemotron_patch
M = "/home/keith/lsa/models/Nemotron-3-Nano-30B-A3B-BF16"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
ids_all = tok(open("/home/keith/lsa/repo/experiments/data/texts/prose-32768.txt").read(), add_special_tokens=False, return_tensors="pt")["input_ids"]
print("prompt tokens available:", ids_all.shape[1], flush=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
print("experts impl:", getattr(model.config, "_experts_implementation", None), flush=True)
GiB = 2**30
def run(n):
    ids = ids_all[:, :n].cuda()
    with torch.no_grad():
        h = model.model(input_ids=ids, use_cache=True).last_hidden_state
        lg = model.lm_head(h[:, -1:])
    return h, lg
def timed(n):
    ids = ids_all[:, :n].cuda(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    with torch.no_grad(): out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    print(f"  prefill {n}: {dt:.2f}s = {n/dt:.0f} tok/s peak {torch.cuda.max_memory_allocated()/GiB:.1f} GiB (weights {WEIGHTS:.1f}) finite={bool(torch.isfinite(out.logits).all())}", flush=True)
WEIGHTS = torch.cuda.memory_allocated() / GiB
# 1. reference (unmodified fallback)
t = time.perf_counter(); h0, lg0 = run(2048); torch.cuda.synchronize(); print(f"fallback 2K: {time.perf_counter()-t:.1f}s", flush=True)
h0, lg0 = h0.float().cpu(), lg0.float().cpu()
# 2. patched
nemotron_patch.apply()
for n in (2048, 4096): run(n)  # autotune warmup
h1, lg1 = run(2048); h1, lg1 = h1.float().cpu(), lg1.float().cpu()
cos_h = F.cosine_similarity(h0.double().flatten(), h1.double().flatten(), dim=0).item()
cos_pos = F.cosine_similarity(h0[0].double(), h1[0].double(), dim=-1)
print("worst 5 positions:", [(i.item(), round(cos_pos[i].item(),4)) for i in cos_pos.argsort()[:5]], "mean per-pos cos", cos_pos.mean().item(), "frac<0.999:", (cos_pos<0.999).float().mean().item())
print("rel err (|d|/|ref|, fro):", ((h0-h1).norm()/h0.norm()).item())
cos_l = F.cosine_similarity(lg0.double().flatten(), lg1.double().flatten(), dim=0).item()
print(f"hidden states: cos={cos_h:.6f} (min per-position cos {cos_pos.min().item():.6f}) max_abs_diff={(h0-h1).abs().max().item():.4g} (ref max abs {h0.abs().max().item():.3g})")
print(f"final-position logits: cos={cos_l:.6f} max_abs_diff={(lg0-lg1).abs().max().item():.4g} (ref max abs {lg0.abs().max().item():.3g}) argmax same={lg0.argmax().item()==lg1.argmax().item()}")
print("VERIFY", "PASS" if cos_h >= 0.999 and cos_l >= 0.999 else "FAIL", flush=True)
print("timing (patched):")
for n in (2048, 8192, 16384, 32768):
    try: timed(n)
    except Exception as e: print(f"  prefill {n}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); torch.cuda.empty_cache()
