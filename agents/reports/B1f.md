# B1f: coherent code and technical prompts

Status: DRAFT, 2026-11-03. Work in progress; this will be updated with final
numbers and the verify.sh output.

## Continuation

The earlier attempt left no `agents/reports/B1f.md` and no
`experiments/data/manifest.coherent.compact.json`. The last acceptance run
printed only:

```
E-001h rocSHMEM Could not open libnuma. Returning NUMAWrapper@src/gda/numa_wrapper.cpp:48
FAILS: missing /ai/github/lsa-rocm/experiments/data/manifest.coherent.compact.json
```

So the work below is new, not a repair of existing outputs.

## Plan

New script `experiments/data/build_coherent.py` (seeded, CPU only, no GPU).
It writes `experiments/data/texts/<id>.txt` and
`experiments/data/manifest.coherent.compact.json`.

Sources (all local and read-only):

- Code, one repository per prompt, source files concatenated in repository
  path order (a module and its neighbours stay contiguous); generated,
  minified, vendored and third_party files are excluded:
  - `code-coh-4096`: stable-diffusion.cpp (MIT), `src/`
  - `code-coh-32768`: llama.cpp (MIT), `src/`, `common/`, `include/`
  - `code-coh-131072`: vllm-0.32 (Apache-2.0), `vllm/`
- Technical, whole documents from one project's documentation set
  (radiance-m2xfp, Apache-2.0, `src/docs/`):
  - `tech-coh-4096`: TOOLS.md (1 source)
  - `tech-coh-32768`: ARCHITECTURES.md (1 source)
  - `tech-coh-131072`: GUIDE.md + OPS.md (2 sources)

Token counts are exact with the Ornith-1.5-9B tokenizer
(`/ai/models/llm/Ornith-1.5-9B`) and `add_special_tokens=False`. The cut is
made by decoding a token prefix and re-encoding until the count is exactly
the target.

## Status

- [ ] script written
- [ ] texts + manifest written
- [ ] verify.sh exits 0

STATUS: DRAFT
