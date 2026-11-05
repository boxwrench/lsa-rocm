import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"); sys.path.insert(0, "/home/keith/lsa/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
M = "/home/keith/lsa/models/Nemotron-3-Nano-30B-A3B-BF16"; N = 8192
tok = AutoTokenizer.from_pretrained(M); ids = tok(open("/home/keith/lsa/texts/prose-coh-524288.txt").read()[:200000], add_special_tokens=False, return_tensors="pt")["input_ids"][:, :N].cuda()
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
mode = sys.argv[1]
if mode == "patched":
    from lsa_rocm.kernels import nemotron_patch; nemotron_patch.apply()
def run(chunk):
    c = DynamicCache(config=model.config)
    with torch.no_grad():
        for s in range(0, N, chunk):
            o = model(input_ids=ids[:, s:s+chunk], position_ids=torch.arange(s, min(s+chunk, N), device="cuda")[None], past_key_values=c, use_cache=True, logits_to_keep=1)
    return o.logits[0, -1].float().cpu()
ref = run(N)
for ch in (4096, 2048, 512):
    l = run(ch); print(mode, "chunk", ch, "cos", F.cosine_similarity(ref, l, dim=0).item(), "maxabs", (ref - l).abs().max().item(), "argmax same", ref.argmax().item() == l.argmax().item(), flush=True)
print("CHECK DONE", flush=True)
