# -*- coding: utf-8 -*-
"""
eval_matrix.py — Retrieval quality eval harness for the Modanisa RAG pipeline
==============================================================================

Runs a small labeled set of test queries through `chat()` (imported from
app.py) and prints a pass/fail matrix across several independent criteria:

    - got_results     : did the query return at least one product?
    - gender_ok        : if a gender was expected, do ALL returned products
                          match it (or "unisex")?
    - category_ok      : if a category_group was expected, does the resolved
                          category_group match?
    - color_ok         : if a color was expected, was it matched exactly
                          (meta["color_matched"] is True) rather than a
                          closest-match fallback?
    - top1_relevant    : is the top result's name a case-insensitive
                          substring/keyword hit against `expect_keywords`?
    - latency_ok       : did the whole chat() call finish under `max_latency_s`?

Each row in EVAL_CASES is one test query. Add more cases as you find bugs —
this is meant to grow into a regression suite, not stay at N=8.

Usage:
    python eval_matrix.py                 # run everything, print matrix
    python eval_matrix.py --case 3        # run a single case (0-indexed)
    python eval_matrix.py --verbose       # show chat()'s own debug prints too
    python eval_matrix.py --save out.csv  # also dump the matrix to CSV

Requires app.py in the same directory (this script imports `chat` and a few
helpers from it — importing app.py has side effects: it loads the CLIP
model, connects to Qdrant, and makes one live Groq call, exactly like
running app.py directly).
"""

import argparse
import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import app  # noqa: E402  (import after argparse is fine; side effects expected)


# ══════════════════════════════════════════════════════════════════════════
# 1. LABELED EVAL SET — edit/extend this as you find real bugs
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    name: str
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None      # local file path, or None
    expect_gender: Optional[str] = None           # "women" | "men" | "kids" | None
    expect_category_group: Optional[str] = None   # e.g. "shoes", "dresses"
    expect_color: Optional[str] = None             # e.g. "black" (catalog vocab)
    expect_keywords: list = field(default_factory=list)  # any-match on top1 name
    max_latency_s: float = 12.0
    gender_filter: Optional[str] = None            # explicit UI-style override
    category_filter: Optional[str] = None
    k: int = 10


EVAL_CASES = [
    EvalCase(
        name="women_dress_color",
        query_text="black modest maxi dress",
        expect_gender="women",
        expect_category_group="dresses",
        expect_color="black",
        expect_keywords=["dress"],
    ),
    EvalCase(
        name="mens_shoes_category_hard_filter",
        query_text="men's black leather shoes",
        expect_gender="men",
        expect_category_group="shoes",
        expect_color="black",
        expect_keywords=["shoe", "sneaker", "boot", "loafer"],
    ),
    EvalCase(
        name="bags_not_dresses_regression",
        # Regression case for the shoes/bags-return-dresses bug described
        # in the module docstring — this MUST come back category_group="bags".
        query_text="brown leather handbag",
        expect_category_group="bags",
        expect_keywords=["bag", "handbag", "tote", "clutch"],
    ),
    EvalCase(
        name="kids_explicit",
        query_text="girls floral dress",
        expect_gender="kids",
        expect_category_group="dresses",
        expect_keywords=["dress"],
    ),
    EvalCase(
        name="unresolvable_color_fallback",
        # "sage green" never appears verbatim in the catalog — should
        # normalize to green/sea green and still return results, not zero.
        query_text="sage green cardigan",
        expect_category_group="outerwear",
        expect_keywords=["cardigan"],
    ),
    EvalCase(
        name="arabic_query",
        query_text="فستان أسود طويل",
        expect_gender="women",
        expect_category_group="dresses",
        expect_color="black",
        expect_keywords=["dress"],
    ),
    EvalCase(
        name="budget_filter",
        query_text="hijab under $15",
        expect_category_group="hijab",
        expect_keywords=["hijab", "scarf", "shawl"],
    ),
    EvalCase(
        name="explicit_gender_ui_override",
        # Simulates the UI's gender dropdown forcing a HARD filter even
        # though the text itself is gender-neutral.
        query_text="cotton t-shirt",
        gender_filter="women",
        expect_gender="women",
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# 2. SCORING — run one case, compute each criterion, return a result row
# ══════════════════════════════════════════════════════════════════════════

def _load_image(path: Optional[str]):
    if not path:
        return None
    from PIL import Image
    return Image.open(path).convert("RGB")


def run_case(case: EvalCase, verbose: bool = False) -> dict:
    row = {"case": case.name, "error": ""}

    query_image = _load_image(case.query_image_path)

    t0 = time.time()
    buf = io.StringIO()
    try:
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
        with ctx:
            resp = app.chat(
                query_text=case.query_text,
                query_image=query_image,
                k=case.k,
                category_filter=case.category_filter,
                gender_filter=case.gender_filter,
                verbose=verbose,
            )
    except Exception as e:
        row.update({
            "got_results": False, "gender_ok": None, "category_ok": None,
            "color_ok": None, "top1_relevant": None, "latency_ok": None,
            "latency_s": round(time.time() - t0, 2), "n_results": 0,
            "error": f"{type(e).__name__}: {e}",
        })
        return row

    elapsed = time.time() - t0
    products = resp.products
    meta = resp.meta

    # got_results
    got_results = len(products) > 0
    row["got_results"] = got_results
    row["n_results"] = len(products)

    # gender_ok — every returned product must match expected gender (or be unisex)
    if case.expect_gender:
        if not got_results:
            row["gender_ok"] = False
        else:
            row["gender_ok"] = all(
                app._row_gender(p) in (case.expect_gender, "unisex") for p in products
            )
    else:
        row["gender_ok"] = None  # not evaluated

    # category_ok — resolved category_group must match expected
    if case.expect_category_group:
        row["category_ok"] = (meta.get("category_group") == case.expect_category_group)
    else:
        row["category_ok"] = None

    # color_ok — must be an exact catalog match, not a closest-match fallback
    if case.expect_color:
        row["color_ok"] = bool(got_results and meta.get("color_matched") is True)
    else:
        row["color_ok"] = None

    # top1_relevant — keyword hit on the top result's name
    if case.expect_keywords:
        if not got_results:
            row["top1_relevant"] = False
        else:
            top1_name = str(products[0].get("name", "")).lower()
            row["top1_relevant"] = any(kw.lower() in top1_name for kw in case.expect_keywords)
    else:
        row["top1_relevant"] = None

    # latency_ok
    row["latency_s"] = round(elapsed, 2)
    row["latency_ok"] = elapsed <= case.max_latency_s

    return row


# ══════════════════════════════════════════════════════════════════════════
# 3. MATRIX RENDERING
# ══════════════════════════════════════════════════════════════════════════

_CRITERIA = ["got_results", "gender_ok", "category_ok", "color_ok",
             "top1_relevant", "latency_ok"]


def _fmt(v) -> str:
    if v is None:
        return "·"       # not evaluated for this case
    return "✓" if v else "✗"


def print_matrix(results: list):
    df = pd.DataFrame(results)

    print("\n" + "═" * 78)
    print("  EVAL MATRIX")
    print("═" * 78)

    header = f"{'case':<32}" + "".join(f"{c:<15}" for c in _CRITERIA) + "n_res  lat(s)"
    print(header)
    print("─" * len(header))

    for r in results:
        line = f"{r['case']:<32}"
        for c in _CRITERIA:
            line += f"{_fmt(r.get(c)):<15}"
        line += f"{r.get('n_results', 0):<7}{r.get('latency_s', '?'):<6}"
        print(line)
        if r.get("error"):
            print(f"    ⚠️  {r['error']}")

    print("─" * len(header))

    # Per-criterion accuracy (only over cases where that criterion was evaluated)
    print("\nPer-criterion accuracy (over evaluated cases only):")
    for c in _CRITERIA:
        vals = [r.get(c) for r in results if r.get(c) is not None]
        if not vals:
            print(f"  {c:<16}: n/a (no cases evaluate this)")
            continue
        acc = sum(1 for v in vals if v) / len(vals)
        print(f"  {c:<16}: {acc:.0%}  ({sum(1 for v in vals if v)}/{len(vals)})")

    n_errors = sum(1 for r in results if r.get("error"))
    if n_errors:
        print(f"\n⚠️  {n_errors}/{len(results)} case(s) raised an exception — see above.")

    print("═" * 78 + "\n")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 4. ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run the RAG eval matrix.")
    parser.add_argument("--case", type=int, default=None,
                         help="Run only this case index (0-based).")
    parser.add_argument("--verbose", action="store_true",
                         help="Show chat()'s own debug prints for each case.")
    parser.add_argument("--save", type=str, default=None,
                         help="Also write the matrix to this CSV path.")
    args = parser.parse_args()

    cases = EVAL_CASES if args.case is None else [EVAL_CASES[args.case]]

    results = []
    for case in cases:
        print(f"▶ Running: {case.name} ...")
        results.append(run_case(case, verbose=args.verbose))

    df = print_matrix(results)

    if args.save:
        df.to_csv(args.save, index=False)
        print(f"✓ Saved matrix to {args.save}")


if __name__ == "__main__":
    main()