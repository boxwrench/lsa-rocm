# lsa-rocm

An exploratory, public lab notebook: does Latent Sparse Attention (LSA) style key retrieval work on current
open models, and is it worth running on AMD ROCm hardware?

**Status: first results are in (preliminary, see [docs/FINDINGS.md](docs/FINDINGS.md)). They measure attention-retrieval quality and index-stage cost, not end-task accuracy.**
This is independent exploration, not a reproduction of the original paper and not an official port.
The idea and reference implementation come from Prefrontal AI's
[latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) repository. Credit and any
correctness questions about the method belong there. This project only asks whether the retrieval step holds up on
the full-attention layers of recent models, and whether it is cheap on an AMD GPU.

## How to read this repo

- [docs/plan.md](docs/plan.md): the question, models, stop/go gates and the batches of work.
- [docs/metrics.md](docs/metrics.md): exactly how retrieval quality is measured.
- `experiments/NNN-name/`: one folder per experiment with hypothesis, command, result and conclusion.
  Negative results are kept.
- [docs/FINDINGS.md](docs/FINDINGS.md): current conclusions, each linked to an experiment, with caveats.
- [docs/fidelity.md](docs/fidelity.md): how this project's LSA compares with the paper's, what matches, what differs, and what that means for the results.
- [docs/long-context-memory.md](docs/long-context-memory.md): a hypothesis, with estimates and open questions, on combining LSA, quantized KV and a CPU-resident cache for much longer context on the same VRAM.

## How the work is done

Much of the implementation and measurement is delegated to LLM agents (DeepSeek models through opencode,
reviewed and directed from Claude Code). Agent reports are kept in `agents/reports/` after review. Numbers in
results come from machine-written JSON files that record the commit, model, software versions and seed, not from
hand copying. Treat anything not backed by an experiment folder as an unverified note.

## Layout

`docs/` plan and definitions, `src/lsa_rocm/` shared code, `tests/`, `experiments/`, `results/` run JSON,
`third_party/` upstream pinned and unmodified (not committed, cloned by script), `agents/reports/` reviewed reports.

## License

Not yet chosen. Until one is added, all rights are reserved.
