#!/usr/bin/env python3
"""Aggregate the B2.0 record files into the summary JSON the report quotes.

Reads one or more ``run.py`` outputs and writes per (dump, index_mode, method or
matched_to) means of the metrics, plus best-config selection and per-layer spread.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def group_key(record):
    method = record["method"]
    if method in ("flat", "tree"):
        return method, method
    return record["matched_to"], method


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()

    summaries = {}
    for name in args.inputs:
        data = json.loads(Path(name).read_text())
        records = data["records"]
        buckets = defaultdict(list)
        best = {}
        per_layer = defaultdict(lambda: defaultdict(list))
        for r in records:
            grp, method = group_key(r)
            key = (r["index_mode"], grp, method)
            buckets[key].append(r)
            per_layer[(r["index_mode"], r["layer"])][(grp, method)].append(r["mass_captured"])
            if method in ("flat", "tree"):
                cfg_key = (
                    r["index_mode"], method, r["config"]["bucket_size"],
                    r["config"]["top_p"], r["config"]["top_k"],
                )
                best.setdefault(cfg_key, []).append(r["mass_captured"])
        summary = {}
        for key, rows in sorted(buckets.items()):
            n = len(rows)
            summary["|".join(map(str, key))] = {
                "n": n,
                "mass_captured": sum(r["mass_captured"] for r in rows) / n,
                "mass_excluding_sinks": sum(r["mass_excluding_sinks"] for r in rows) / n,
                "recall_at_k": sum(r["recall_at_k"] for r in rows) / n,
                "output_error": sum(r["output_error"] for r in rows) / n,
                "budget_fraction": sum(r["budget_fraction"] for r in rows) / n,
                "index_build_s": sum(r["index_build_s"] for r in rows) / n,
            }
        sweeps = {
            "|".join(map(str, key)): {
                "n": len(values),
                "mass_captured": sum(values) / len(values),
            }
            for key, values in sorted(best.items())
        }
        layer_rows = {
            f"{mode}|{layer}": {
                f"{grp}|{method}": sum(v) / len(v)
                for (grp, method), v in methods.items()
            }
            for (mode, layer), methods in sorted(per_layer.items())
        }
        summaries[Path(name).name] = {
            "meta": data["meta"],
            "summary": summary,
            "sweep": sweeps,
            "per_layer": layer_rows,
        }
    Path(args.out).write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
