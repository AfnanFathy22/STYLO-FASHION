# -*- coding: utf-8 -*-
"""
Read-only diagnostic — no writes. Directly queries the local Qdrant
collection to check whether `category_group` / `gender` payload fields
actually exist and match, bypassing the vector-search + LLM layers
entirely. Run this to compare against what worked in Colab.

    python diagnose.py
"""
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app import qdrant, MM_COLLECTION, COLLECTION

print(f"\nqdrant-client info:")
try:
    from importlib.metadata import version as _pkg_version
    print(f"  version: {_pkg_version('qdrant-client')}")
except Exception as e:
    print(f"  (couldn't determine version: {e})")

for name in (COLLECTION, MM_COLLECTION):
    info = qdrant.get_collection(name)
    print(f"\nCollection '{name}':")
    print(f"  points_count: {info.points_count}")

print(f"\n--- Sample of 5 raw points from '{MM_COLLECTION}' ---")
points, _ = qdrant.scroll(
    collection_name=MM_COLLECTION, limit=5, with_payload=True, with_vectors=False
)
for p in points:
    payload = dict(p.payload or {})
    print(
        f"  id={p.id}  category_group={payload.get('category_group')!r}  "
        f"gender={payload.get('gender')!r}  category={payload.get('category')!r}  "
        f"name={str(payload.get('name'))[:40]!r}"
    )

print(f"\n--- Exact count where category_group == 'dresses' ---")
cnt = qdrant.count(
    collection_name=MM_COLLECTION,
    count_filter=Filter(
        must=[FieldCondition(key="category_group", match=MatchValue(value="dresses"))]
    ),
    exact=True,
)
print(f"  count: {cnt.count}")

print(f"\n--- Breakdown of category_group values across the whole collection ---")
from collections import Counter
counts = Counter()
offset = None
scanned = 0
while True:
    pts, offset = qdrant.scroll(
        collection_name=MM_COLLECTION, limit=500, offset=offset,
        with_payload=["category_group"], with_vectors=False,
    )
    for p in pts:
        counts[(p.payload or {}).get("category_group")] += 1
    scanned += len(pts)
    if offset is None or not pts:
        break
print(f"  scanned {scanned} points")
for val, c in counts.most_common(20):
    print(f"  {val!r}: {c}")