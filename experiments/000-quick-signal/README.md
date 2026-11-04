# 000-quick-signal

**Status:** done, preliminary, **superseded by 002.** Its numbers include the recent-key window and sinks in the captured mass, which inflates them, and it does not follow the paper's index recipe (see docs/paper-notes.md). With the paper's metric (mass outside the window) retrieval is far weaker, see experiment 002.

## Hypothesis
LSA-style retrieval on the full-attention layers of a recent hybrid model captures much more attention mass than
a random or a most-recent selection of the same size.

## Setup
Ornith 1.5 9B, 8 full-attention layers, real prose (prose-4096), 4096 tokens, the last 32 queries, 4 KV heads per
layer. One default configuration, about 4.7% of keys selected (64 retrieved + 128 recent + 4 sinks). Mass is scored
against the true attention of every query head in the KV group. Controls are matched in size to each method:
random, most recent keys, and the exact top keys (an upper bound). Group queries are summed then normalised, which
approximates the upstream recipe. Metric definitions: [docs/metrics.md](../../docs/metrics.md).

## Command
Dump: `python -m lsa_rocm.dump --model <Ornith-1.5-9B> --prompt-file experiments/data/texts/prose-4096.txt --max-tokens 4096 --n-queries 32 --out results/dumps/ornith-prose-4096 --device cuda`
Analysis: `python quick_signal.py results/dumps/ornith-prose-4096 results.json` (CPU torch, `lsa` on the path).

## Result (full-dims index; mean over the 8 layers, from `results.json`)

| Selection at the same size | Mass captured |
|---|---|
| Random keys | 0.05 |
| Most recent keys | 0.44 |
| LSA flat or tree | 0.60 |
| Exact top keys (upper bound) | 0.83 |

Retrieval beats random everywhere and beats recency in every layer (layer 27: 0.48 vs 0.27; layer 3 only 0.43 vs
0.40). Tree and flat agree within 0.005. Position-free dims (dropping the rotated 64) differ from full dims by under 0.01.

## Conclusion
The idea is alive on real attention at 4K. It gets roughly 60 to 78% of what perfect top-k would capture at this budget.

## Caveats
4K only, one document, one default configuration, queries only from the end of the prompt, no tuning, hierarchy
barely exercised at this size. Group-summed queries are an approximation. Treat the numbers as indicative.
