# -*- coding: utf-8 -*-
"""
One-off maintenance: diagnose and fix a Qdrant collection that was indexed
BEFORE the `category_group` (and/or `gender`) payload fields existed on each
point. `chat()` uses a HARD Qdrant filter on these fields — a point with no
such field can never match, for ANY category/gender, which is exactly the
"Couldn't find any matches (category locked to 'dresses')" symptom even
though your collection clearly has dresses in it.

Run this once, right after copying a Qdrant collection in from Drive/an
older indexing run:

    python backfill.py

It runs a DRY RUN first (no writes) so you can see how many points are
missing each field before committing to anything.
"""
from app import (
    MM_COLLECTION,
    backfill_missing_category_groups,
    backfill_missing_genders,
    ensure_category_group_index,
)

print(f"\nTarget collection: {MM_COLLECTION}\n")

print("=== DRY RUN: category_group ===")
cg_summary = backfill_missing_category_groups(dry_run=True)
print(cg_summary)

print("\n=== DRY RUN: gender ===")
g_summary = backfill_missing_genders(dry_run=True)
print(g_summary)

if cg_summary["updated"] == 0 and g_summary["updated"] == 0:
    print(
        "\n✅ Nothing to backfill — every point already has both fields set.\n"
        "   If searches are still returning zero matches, the category/gender\n"
        "   you're filtering on may genuinely have no products in this\n"
        "   collection — try 'All categories' / 'Auto-detect' to confirm."
    )
    raise SystemExit(0)

print(
    f"\n{cg_summary['updated']} point(s) need a category_group backfill, "
    f"{g_summary['updated']} point(s) need a gender backfill."
)
answer = input("\nProceed with writing these updates? [y/N] ").strip().lower()

if answer == "y":
    print("\n=== WRITING: category_group ===")
    print(backfill_missing_category_groups(dry_run=False))
    ensure_category_group_index()

    print("\n=== WRITING: gender ===")
    print(backfill_missing_genders(dry_run=False))

    print(
        "\n✅ Done. The Qdrant collection itself is now updated on disk — "
        "just restart app.py (no need to re-run this script again unless "
        "you add new un-tagged points later)."
    )
else:
    print("Aborted — no changes written.")