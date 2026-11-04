#!/usr/bin/env python3
"""Build evaluation prompts of real long text at 4K, 32K and 128K tokens.

All sources are read-only files already on disk under /ai/huggingface (and
/ai/datasets, which is empty on this machine).  No GPU is used.

Genres and sources
------------------
prose     sgoel9/paul_graham_essays (MIT)
code      openai/openai_humaneval (MIT),
          HF transformers/diffusers module source files (Apache-2.0),
          fenced code blocks from model-card READMEs (per-repo licence)
technical TIGER-Lab/MMLU-Pro (MIT), openai/gsm8k (MIT), google/IFEval (Apache-2.0)

The Ornith-1.5-9B tokenizer (/ai/models/llm/Ornith-1.5-9B) is used with
add_special_tokens=False.  Each prompt is assembled at the token level from
seeded-shuffled real documents, cut to exactly the target token count, decoded
and then re-encoded to verify the count.

Outputs
-------
experiments/data/texts/<id>.txt   plain-text prompts (gitignored)
experiments/data/manifest.json    id, source path, genre, licence, token count, sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "experiments", "data")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

TOKENIZER_PATH = "/ai/models/llm/Ornith-1.5-9B"
HF_ROOT = "/ai/huggingface"

SEED = 20261103
LENGTHS = (4096, 32768, 131072)
GENRES = ("prose", "code", "technical")

# Import the tokenizer before any optional pyarrow fallback path is appended,
# so the named comfyui-rocm714 environment's tokenizers is the one used.
from tokenizers import Tokenizer  # noqa: E402

_TOKENIZER = Tokenizer.from_file(os.path.join(TOKENIZER_PATH, "tokenizer.json"))


def _encode(text: str) -> List[int]:
    return _TOKENIZER.encode(text, add_special_tokens=False).ids


def _decode(ids: List[int]) -> str:
    return _TOKENIZER.decode(ids, skip_special_tokens=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pyarrow():
    """Return pyarrow.parquet, falling back to an existing env's pyarrow.

    The named comfyui-rocm714 environment has tokenizers but not pyarrow.
    kitchen-test is an existing environment with pyarrow 25.0.1; its site-packages
    is appended (not prepended) so the named env's tokenizers keeps precedence.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq
    except ImportError:
        fallback = "/ai/environments/kitchen-test/lib/python3.12/site-packages"
        if os.path.isdir(fallback) and fallback not in sys.path:
            sys.path.append(fallback)
        import pyarrow.parquet as pq  # type: ignore

        return pq


def _read_parquet(path: str) -> List[dict]:
    pq = _pyarrow()
    return pq.read_table(path).to_pylist()


# ---------------------------------------------------------------------------
# Source loaders.  Each yields dicts with text, source_path, licence, document_id.
# ---------------------------------------------------------------------------

PG_ESSAYS = (
    "/ai/huggingface/hub/datasets--sgoel9--paul_graham_essays/snapshots/"
    "0c7155a53c25e24c9b9858314460e0ee1f5c3e4e/pual_graham_essays.csv"
)
HUMANEVAL = (
    "/ai/huggingface/hub/datasets--openai--openai_humaneval/snapshots/"
    "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet"
)
MMLUPRO_TEST = (
    "/ai/huggingface/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/"
    "b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet"
)
GSM8K_TRAIN = (
    "/ai/huggingface/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
)
IFEVAL = (
    "/ai/huggingface/hub/datasets--google--IFEval/snapshots/"
    "966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl"
)


def load_pg_essays() -> Iterable[dict]:
    with open(PG_ESSAYS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text") or ""
            if text.strip():
                yield {
                    "text": text,
                    "source_path": PG_ESSAYS,
                    "licence": "MIT",
                    "document_id": "paul_graham_essays/" + str(row.get("id", "")),
                }


def load_humaneval() -> Iterable[dict]:
    for row in _read_parquet(HUMANEVAL):
        text = (
            row["prompt"]
            + row["canonical_solution"]
            + "\n\n"
            + row["test"]
        )
        yield {
            "text": text,
            "source_path": HUMANEVAL,
            "licence": "MIT",
            "document_id": "openai_humaneval/" + str(row["task_id"]),
        }


def load_mmlupro() -> Iterable[dict]:
    for row in _read_parquet(MMLUPRO_TEST):
        options = "\n".join(
            f"({chr(65 + j)}) {opt}" for j, opt in enumerate(row["options"])
        )
        text = (
            row["question"]
            + "\n"
            + options
            + "\n\nAnswer: "
            + str(row["answer"])
        )
        if row.get("cot_content"):
            text += "\n\n" + row["cot_content"]
        yield {
            "text": text,
            "source_path": MMLUPRO_TEST,
            "licence": "MIT",
            "document_id": f"mmlu-pro/{row['question_id']}",
        }


def load_gsm8k() -> Iterable[dict]:
    for i, row in enumerate(_read_parquet(GSM8K_TRAIN)):
        text = row["question"] + "\n\n" + row["answer"]
        yield {
            "text": text,
            "source_path": GSM8K_TRAIN,
            "licence": "MIT",
            "document_id": f"gsm8k/{i}",
        }


def load_ifeval() -> Iterable[dict]:
    with open(IFEVAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield {
                "text": row["prompt"],
                "source_path": IFEVAL,
                "licence": "Apache-2.0",
                "document_id": "ifeval/" + str(row.get("key", "")),
            }


def load_hf_modules_code() -> Iterable[dict]:
    seen = set()
    base = os.path.join(HF_ROOT, "modules")
    for root, _dirs, files in os.walk(base):
        if "/venv/" in root or "/site-packages/" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.realpath(os.path.join(root, name))
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            if not text.strip():
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            yield {
                "text": text,
                "source_path": path,
                "licence": "Apache-2.0",
                "document_id": "hf_modules/" + digest[:12],
            }


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)


def _frontmatter_licence(text: str) -> str:
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return "unknown"
    fm = m.group(1)
    m2 = re.search(r"^license:\s*([^\n]+)", fm, re.M)
    if m2:
        return m2.group(1).strip()
    m3 = re.search(r"^license:\s*\n((?:\s*-\s*\S+[ \t]*\n)+)", fm, re.M)
    if m3:
        return ", ".join(
            line.strip().lstrip("-").strip() for line in m3.group(1).splitlines()
        )
    return "unknown"


def load_readme_code() -> Iterable[dict]:
    seen = set()
    for base_name in ("hub", "modules", "datasets"):
        base = os.path.join(HF_ROOT, base_name)
        for root, _dirs, files in os.walk(base):
            if "/venv/" in root or "/site-packages/" in root:
                continue
            for name in files:
                if not name.lower().endswith(".md"):
                    continue
                path = os.path.realpath(os.path.join(root, name))
                try:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                licence = _frontmatter_licence(text)
                for m in _FENCE_RE.finditer(text):
                    lang = m.group(1).lower()
                    code = m.group(2)
                    if len(code.strip()) < 20:
                        continue
                    if len(_encode(code)) < 5:
                        continue
                    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    yield {
                        "text": code,
                        "source_path": path,
                        "licence": licence,
                        "document_id": "readme_code/" + digest[:12],
                        "language": lang,
                    }


def build_pools() -> Dict[str, List[dict]]:
    pools: Dict[str, List[dict]] = {g: [] for g in GENRES}
    for doc in load_pg_essays():
        pools["prose"].append(doc)
    for doc in load_humaneval():
        pools["code"].append(doc)
    for doc in load_hf_modules_code():
        pools["code"].append(doc)
    for doc in load_readme_code():
        pools["code"].append(doc)
    for doc in load_mmlupro():
        pools["technical"].append(doc)
    for doc in load_gsm8k():
        pools["technical"].append(doc)
    for doc in load_ifeval():
        pools["technical"].append(doc)
    return pools


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _stable_cut(ids: List[int], target: int) -> Tuple[int, str, List[int]]:
    """Find a cut point k whose decoded text re-encodes to exactly target tokens.

    BPE decode/encode is not always an exact inverse at an arbitrary token
    boundary (a trailing partial word or space can re-tokenize differently), so
    we search a small window around the target.  The returned text is always a
    prefix of the concatenated real documents.
    """
    if len(ids) < target:
        raise RuntimeError(
            f"pool has {len(ids)} tokens, need {target}; cannot fill this prompt"
        )
    for delta in range(0, 2001):
        if delta == 0:
            candidates = (target,)
        else:
            candidates = (target + delta, target - delta)
        for k in candidates:
            if k < 1 or k > len(ids):
                continue
            text = _decode(ids[:k])
            check = _encode(text)
            if len(check) == target:
                return k, text, check
    raise RuntimeError(
        f"no token cut in [{target - 2000}, {target + 2000}] re-encodes to {target}"
    )


def assemble(docs: List[dict], target: int, rng: random.Random) -> Tuple[str, List[dict], int]:
    order = list(range(len(docs)))
    rng.shuffle(order)
    sep = _encode("\n\n")
    ids: List[int] = []
    spans: List[Tuple[int, int, int]] = []
    # Collect a little past the target so the stable-cut search has room.
    limit = target + 2000
    for idx in order:
        doc_ids = _encode(docs[idx]["text"])
        if not doc_ids:
            continue
        if ids:
            ids.extend(sep)
        start = len(ids)
        ids.extend(doc_ids)
        spans.append((idx, start, len(ids)))
        if len(ids) >= limit:
            break
    k, text, check = _stable_cut(ids, target)
    if len(check) != target:
        raise RuntimeError(
            f"tokenizer round-trip gave {len(check)} tokens, expected {target}"
        )
    used = []
    for idx, start, end in spans:
        if start >= k:
            break
        doc = docs[idx]
        used.append(
            {
                "document_id": doc["document_id"],
                "source_path": doc["source_path"],
                "licence": doc["licence"],
                "tokens": min(end, k) - start,
            }
        )
    return text, used, k


def _licence_summary(used: List[dict]) -> str:
    licences = sorted({d["licence"] for d in used})
    if len(licences) == 1:
        return licences[0]
    return "mixed (" + ", ".join(licences) + ")"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verify", action="store_true", help="verify existing outputs and exit")
    args = parser.parse_args()

    os.makedirs(TEXT_DIR, exist_ok=True)

    if args.verify:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
        ok = True
        for p in manifest["prompts"]:
            path = os.path.join(DATA_DIR, p["text_file"])
            with open(path, encoding="utf-8") as f:
                text = f.read()
            n = len(_encode(text))
            digest = _sha256(text)
            good = n == p["token_count"] and digest == p["sha256"]
            ok = ok and good
            print(
                f"{p['id']:<18} tokens={n:<7} expected={p['token_count']:<7} "
                f"sha256={'ok' if digest == p['sha256'] else 'MISMATCH'} "
                f"{'OK' if good else 'FAIL'}"
            )
        print("VERIFY", "OK" if ok else "FAIL")
        return 0 if ok else 1

    pools = build_pools()
    for genre in GENRES:
        total = sum(len(_encode(d["text"])) for d in pools[genre])
        print(
            f"pool {genre:<10} documents={len(pools[genre]):<6} tokens={total}"
        )

    prompts = []
    for genre in GENRES:
        for length in LENGTHS:
            # Deterministic per-(genre, length) shuffle.
            rng = random.Random(args.seed + length + 1000003 * GENRES.index(genre))
            text, used, cut = assemble(pools[genre], length, rng)
            token_count = len(_encode(text))
            if token_count != length:
                raise RuntimeError(
                    f"{genre}-{length}: assembled {token_count} tokens, expected {length}"
                )
            prompt_id = f"{genre}-{length}"
            rel_path = f"texts/{prompt_id}.txt"
            with open(os.path.join(DATA_DIR, rel_path), "w", encoding="utf-8") as f:
                f.write(text)
            source_paths = sorted({d["source_path"] for d in used})
            prompts.append(
                {
                    "id": prompt_id,
                    "genre": genre,
                    "length": length,
                    "source_path": "; ".join(source_paths),
                    "source_paths": source_paths,
                    "licence": _licence_summary(used),
                    "licences": sorted({d["licence"] for d in used}),
                    "token_count": token_count,
                    "sha256": _sha256(text),
                    "text_file": rel_path,
                    "cut": cut,
                    "num_documents": len(used),
                    "documents": used,
                }
            )
            print(
                f"wrote {prompt_id:<18} tokens={token_count:<7} "
                f"docs={len(used):<5} sha256={prompts[-1]['sha256'][:16]}..."
            )

    try:
        import pyarrow  # type: ignore

        pyarrow_info = getattr(pyarrow, "__version__", "unknown")
        pyarrow_path = getattr(pyarrow, "__file__", "unknown")
    except ImportError:
        pyarrow_info = "unavailable"
        pyarrow_path = "unavailable"

    manifest = {
        "generated_by": "experiments/data/build_prompts.py",
        "seed": args.seed,
        "tokenizer": TOKENIZER_PATH,
        "tokenizer_vocab_size": _TOKENIZER.get_vocab_size(),
        "add_special_tokens": False,
        "lengths": list(LENGTHS),
        "genres": list(GENRES),
        "pyarrow_version": pyarrow_info,
        "pyarrow_path": pyarrow_path,
        "prompts": prompts,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
