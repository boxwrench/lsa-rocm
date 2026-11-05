import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/keith/lsa/src")
import transformers.models.nemotron_h.modeling_nemotron_h as mh
from lsa_rocm.kernels import nemotron_patch as np_
from lsa_rocm.kernels.moe import fused_experts_relu2
dev = "cuda"; torch.manual_seed(0)
def rep(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"{name}: cos={cos:.6f} maxabs={(a-b).abs().max().item():.4g} refmaxabs={b.abs().max().item():.3g}", flush=True)
    return cos
H, P, G, N, CS = 64, 64, 8, 128, 128
ok = True
for T in (1000, 2048):  # 1000 tests ragged last chunk
    for init in (False, True):
        x = (torch.randn(1, T, H, P, device=dev) * 0.5).bfloat16()
        dt = torch.randn(1, T, H, device=dev).bfloat16()
        A = -torch.exp(torch.randn(H, device=dev) * 0.5)
        Bm = (torch.randn(1, T, G, N, device=dev) * 0.5).bfloat16(); Cm = (torch.randn(1, T, G, N, device=dev) * 0.5).bfloat16()
        D = torch.randn(H, device=dev); dtb = torch.randn(H, device=dev) * 0.5
        i0 = torch.randn(1, H, P, N, device=dev) * 0.1 if init else None
        kw = dict(chunk_size=CS, D=D, dt_bias=dtb, dt_softplus=True, dt_limit=(0.0, float("inf")), initial_states=i0, return_final_states=True)
        ro, rf = mh.mamba2_chunk_scan.__wrapped__(x, dt, A, Bm, Cm, **kw) if hasattr(mh.mamba2_chunk_scan, "__wrapped__") else mh.mamba2_chunk_scan(x, dt, A, Bm, Cm, **kw)
        to, tf = np_.chunk_scan_triton(x, dt, A, Bm, Cm, **kw)
        ok &= rep(f"ssd T={T} init={init} out", to, ro) > 0.999
        ok &= rep(f"ssd T={T} init={init} final_state", tf, rf) > 0.999
# conv
Cc = 6144
for T in (1000, 2048):
    x = torch.randn(1, T, Cc, device=dev).bfloat16(); w = (torch.randn(Cc, 4, device=dev) * 0.5).bfloat16(); b = torch.randn(Cc, device=dev).bfloat16()
    xt = x.transpose(1, 2)
    r = mh.causal_conv1d_fn(xt, w, b, activation="silu"); t = np_._conv_fn(xt, w, b, activation="silu")
    ok &= rep(f"conv T={T}", t, r) > 0.999
# moe
class Cfg: n_routed_experts=128; hidden_size=2688; moe_intermediate_size=1856; moe_latent_size=None; mlp_hidden_act="relu2"; _experts_implementation="eager"
ex = mh.NemotronHExperts(Cfg()).to(dev).bfloat16()
torch.nn.init.normal_(ex.up_proj, std=0.02); torch.nn.init.normal_(ex.down_proj, std=0.02)
for M in (64, 2048):
    x = torch.randn(M, 2688, device=dev).bfloat16()
    idx = torch.stack([torch.randperm(128, device=dev)[:6] for _ in range(M)]); w = torch.rand(M, 6, device=dev) * 2.5
    r = mh.NemotronHExperts.forward(ex, x, idx, w); t = fused_experts_relu2(x, ex.up_proj, ex.down_proj, idx, w)
    ok &= rep(f"moe M={M}", t, r) > 0.999
print("ALL OK" if ok else "FAILURES")
