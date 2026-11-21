# -*- coding: utf-8 -*-
"""
Modanisa Fashion RAG — Complete Pipeline
=========================================

A hybrid text+image retrieval-augmented shopping assistant for Modanisa.

Pipeline stages:
    1. Environment setup   — installs, Drive mount, config
    2. Data layer           — CSV load + column normalization
    3. Embedding layer      — CLIP-based text/image encoder (modern transformers)
    4. Vector store layer   — Qdrant connection + restore-from-Drive
    5. LLM layer            — Groq helpers (chat, vision, JSON intent parsing)
    6. Query understanding  — keyword fallback parsers + LLM intent extraction
    7. Retrieval pipeline   — chat() orchestrates steps 1-6 into one search
    8. Presentation layer   — ChatResponse (CLI-style), Gradio product cards
    9. Gradio UI            — interactive demo
   10. Image proxy          — server-to-server image fetch (bypasses Modanisa's
                              hotlink/referer protection so <img> tags in a
                              separate frontend, e.g. v0, can load thumbnails)

Run top to bottom in a single Colab session. Do not re-run earlier install
cells mid-session — restart the runtime if you change package versions.

GENDER FILTERING
----------------
Products are tagged women/men/kids/unisex (see `_infer_gender` in the data
layer and the `gender` field in the vector payload). Gender resolution
distinguishes two kinds of signal:

  - EXPLICIT signals (an explicit UI/API `gender_filter`, or a keyword the
    customer actually typed, e.g. "women's", "men's", "kids") are strong
    enough to enforce as a HARD Qdrant `must` filter at query time —
    cross-gender items are excluded from retrieval entirely.

  - INFERRED signals (the LLM's read of an uploaded photo, or a keyword
    match against the vision model's auto-generated image caption) are
    treated as a SOFT, client-side filter only (`_apply_post_filters` /
    `_row_gender`). Vision-based gender guesses are not reliable enough to
    hard-exclude products at the database layer.

See `backfill_missing_genders()` in the vector store layer — indexed points
whose `gender` payload field is blank/missing will never satisfy a Qdrant
`MatchAny` hard filter, so run that once against any collection indexed
before per-item gender tagging existed.

CATEGORY FILTERING  (**fix for shoes/bags returning dresses/shirts**)
----------------------------------------------------------------------
Previously, `category` (from the LLM intent JSON or a UI dropdown) was
ONLY ever applied as a client-side post-filter (`_apply_post_filters` /
`_matches_category`), and that post-filter used `_safe_filter`, which
silently ABANDONS a filter — returning the unfiltered candidate pool —
whenever fewer than 2 rows survive it.

That combination broke badly for categories the CLIP embedding space is
weak at distinguishing (shoes, bags): a query like "black shoes" pulls a
dense-vector top-K dominated by black dresses/tops (which embed more
confidently/are overrepresented), too few real shoe items land in that
pool to satisfy `_safe_filter`'s minimum, the category filter gets
silently skipped, and the customer sees dresses and shirts in a shoes
search.

Fix: a coarse, deterministic `category_group` field (shoes / bags /
dresses / tops / ... — see `CATEGORY_GROUP_RULES`) is computed for every
product at index time, and — whenever a query confidently resolves to one
of these buckets (`_resolve_category_group`) — it's promoted to a HARD
Qdrant `must` filter, exactly like gender_hard. This constrains the dense
candidate pool itself instead of hoping enough of the right category
surfaces in the top-K. Category is *also* resolved via a small keyword
fallback (`extract_category`) so this doesn't solely depend on the LLM
intent call correctly classifying every query.

The old fine-grained `category` field and `_matches_category` client-side
logic are left in place as a secondary, redundant safety net — harmless
now that the DB-level filter already guarantees relevance.

See `backfill_missing_category_groups()` and `ensure_category_group_index()`
in the vector store layer — run these once against any collection indexed
before `category_group` existed.

COLOR FILTERING  (**fix for vision-derived colors like "sage green" returning 0 results**)
--------------------------------------------------------------------------------------------
`color` in `_apply_post_filters` is intentionally strict, no-fallback: if
the customer explicitly asked for a color, showing the wrong one is a
correctness bug, not a "too strict, loosen up" situation (see the comment
above that block). That's the right call *if* the color string is drawn
from the same controlled vocabulary (`COLORS`) that product `colors`/`name`
fields actually use.

The break: `enhance_query()`'s intent LLM (and the vision caption feeding
it) can output open-ended color phrases — "sage green", "dusty olive",
"charcoal" — that never appear verbatim in the catalog, which only ever
uses `COLORS` terms like "green", "sea green", "dark gray". A strict
substring match against a color no product will ever literally contain
zeroes out the whole candidate pool, even when a clearly-matching product
(e.g. a green dress) is sitting right there in the index.

Fix: `normalize_color()` snaps any free-text color (from the LLM, the
vision caption, or keyword extraction) onto the closest `COLORS` vocabulary
term — exact match, then synonym table, then substring-either-direction,
then shared significant word — before it ever reaches the strict filter in
`_apply_post_filters()`. The intent prompt is also given the `COLORS` list
directly so the LLM is nudged to pick from it in the first place, with
`normalize_color()` as a safety net for whatever slips through.

STYLE / CLOSEST-MATCH  (patched)
---------------------------------
Two additional customer intents are now handled:

  - "same dress but shorter / with long sleeves" style edits relative to
    an uploaded photo: `STYLE_MODIFIERS` / `extract_style_modifier()`
    fold a length/sleeve tag into `search_query` (semantic, via CLIP —
    there's no structured payload field for this), and the intent prompt
    now instructs the LLM to rewrite `search_query` to describe the
    *modified* item rather than the item exactly as it appears in the
    photo.

  - When the requested color has zero exact matches, we no longer return
    a dead end. `apply_color_filter_with_fallback()` falls back to the
    closest candidates (same style/fabric/gender/category/price/size —
    just not the requested color) and `meta["color_matched"] = False`
    tells the LLM to say plainly these are the nearest options, not an
    exact hit.

IMAGE PROXY  (**fix for product photos not loading in an external frontend, e.g. v0**)
-----------------------------------------------------------------------------------------
Modanisa's own CDN blocks hotlinked image requests that arrive without a
matching Referer/Origin (typical anti-hotlink protection) — this is why an
`image_url` returned by `/search` loads fine when the *backend* fetches it
but 404s / errors when a browser tries to load it directly in an `<img>`
tag from a different frontend origin (v0, localhost:3000, etc).

Fix: `/image-proxy?url=<encoded original url>` fetches the image
server-to-server (with a Referer Modanisa's CDN accepts) and streams the
bytes back from *this* backend's own origin. `_proxify()` rewrites every
`image_url` / `all_images` entry returned by `/search` and `/search-image`
to point at this proxy endpoint instead of the raw Modanisa URL, so the
frontend never talks to Modanisa's CDN directly.
"""

# ══════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════

# ── 1.1 Installs ─────────────────────────────────────────────────────────
# Run once, locally, inside your venv:
#   pip install -r requirements.txt
#
# NOTE: we do NOT install the `fashion-clip` pip package. Its bundled loading
# code calls transformers internals (`use_auth_token=`, `_is_control`) that
# newer transformers releases removed, and pinning transformers back to an
# old version to satisfy it breaks huggingface_hub/gradio compatibility
# instead. Section 3 below loads the same public checkpoint
# (`patrickjohncyh/fashion-clip`) directly through modern `transformers`,
# which sidesteps that whole dependency chain.
#
# NOTE (image proxy, section 10): also requires `httpx` — add it to
# requirements.txt and run:
#   pip install httpx --break-system-packages

import io
import os
import re
import json
import base64
import shutil
import warnings
import textwrap
from pathlib import Path
from dataclasses import dataclass, field as dc_field
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPModel, CLIPProcessor
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

warnings.filterwarnings("ignore")

# Load GROQ_API_KEY (and any other secrets) from a local .env file placed
# next to this script. See .env.example.
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 1.2 Configuration ───────────────────────────────────────────────────────
# Everything is now relative to this script's folder (BASE_DIR) instead of
# Colab's /content/drive/MyDrive/... — copy your CSV + qdrant_storage folder
# (the one you downloaded from Google Drive) into DATA_DIR below.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CONFIG = {
    # Paths — point these at your local copies of the CSV and Qdrant folder.
    "valid_csv_path": str(DATA_DIR / "modanisa_valid.csv"),
    "qdrant_local":   str(DATA_DIR / "qdrant_storage"),
    # Embedding model
    "clip_model_name": "patrickjohncyh/fashion-clip",
    # Qdrant
    "collection_image": "modanisa_image",
    "collection_mm":    "modanisa_multimodal",
    "score_threshold":  0.20,
    "rag_top_k":        10,
    "fetch_multiplier": 3,
    # Groq / LLM
    # NOTE (Aug 2026): Groq deprecated llama-3.3-70b-versatile and
    # meta-llama/llama-4-scout-17b-16e-instruct. Using their recommended
    # replacements — check https://console.groq.com/docs/deprecations if
    # these ever 404 again, since Groq retires models periodically.
    "llm_model":        "openai/gpt-oss-120b",
    "vision_model":     "qwen/qwen3.6-27b",   # multimodal (gpt-oss-120b is text-only)
    "llm_max_tokens":   512,
    # Misc
    "img_fetch_timeout": 20,
}

# ── 1.3 Verify local Qdrant folder ──────────────────────────────────────────
# No more "restore from Drive" step — you already have the Qdrant storage
# folder locally (copied out of Drive), so we just point straight at it and
# clear any stale lock file left over from a previous Colab/qdrant session.
def prepare_local_qdrant(cfg: dict) -> str:
    local = cfg["qdrant_local"]
    if not (os.path.exists(local) and os.listdir(local)):
        raise FileNotFoundError(
            f"❌ No Qdrant storage found at '{local}'.\n"
            f"   Copy your qdrant_storage folder (downloaded from Google Drive) "
            f"into: {local}"
        )
    print(f"✅ Using local Qdrant storage at {local}")

    lock = os.path.join(local, ".lock")
    if os.path.exists(lock):
        os.remove(lock)
        print("✓ Stale lock removed")
    return local


LOCAL_QDRANT = prepare_local_qdrant(CONFIG)


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA LAYER — CSV load + column normalization
# ══════════════════════════════════════════════════════════════════════════

def _parse_price(val) -> float:
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except Exception:
        return 0.0


def _parse_discount(val) -> float:
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return 0.0


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().upper() == "TRUE"


def _infer_gender(row) -> str:
    t = f"{row.get('name', '')} {row.get('category', '')}".lower()
    if any(w in t for w in ["men", "man", "boy"]):
        return "men"
    if any(w in t for w in ["girl", "kids", "child"]):
        return "kids"
    if any(w in t for w in ["women", "modest", "hijab", "abaya", "dress", "scarf"]):
        return "women"
    return "unisex"


def _build_category(row) -> str:
    """Derive a unified `category` slug from whichever taxonomy columns
    exist. This CSV ships masterCategory/subCategory/articleType instead of
    a single `category` field, so downstream code (top-category summary,
    chat() category filter, db_info()) needs this normalized first."""
    for src in ("articleType", "subCategory", "masterCategory"):
        val = str(row.get(src, "")).strip()
        if val and val.lower() != "nan":
            return val.lower().replace(" ", "-")
    return "other"


# ── Coarse category buckets for HARD filtering ─────────────────────────────
CATEGORY_GROUP_RULES = {
    "shoes":       ["shoe", "sneaker", "trainer", "boot", "sandal", "heel",
                     "flip flop", "flip-flop", "slipper", "loafer", "espadrille"],
    "bags":        ["bag", "backpack", "handbag", "clutch", "wallet", "purse", "tote"],
    "dresses":     ["dress", "gown", "abaya", "jumpsuit"],
    "tops":        ["shirt", "blouse", "top", "tunic", "tee", "t-shirt", "sweatshirt", "hoodie"],
    "outerwear":   ["jacket", "coat", "cardigan", "blazer", "vest"],
    "bottoms":     ["trouser", "jean", "pant", "skirt", "short"],
    "hijab":       ["hijab", "scarf", "shawl"],
    "swimwear":    ["swim", "bikini", "beachwear"],
    "accessories": ["accessor", "jewel", "belt", "watch", "sunglass", "hat", "cap"],
    "suits":       ["suit", "blazer set", "co-ord"],
}
_CATEGORY_GROUP_ORDER = ["swimwear", "shoes", "bags", "hijab", "outerwear",
                          "suits", "dresses", "bottoms", "tops", "accessories"]


def _infer_category_group(row) -> str:
    text = " ".join(str(row.get(f, "")) for f in
                     ("articleType", "subCategory", "masterCategory",
                      "category", "name")).lower()
    for group in _CATEGORY_GROUP_ORDER:
        if any(kw in text for kw in CATEGORY_GROUP_RULES[group]):
            return group
    return "other"


def _resolve_category_group(category_text):
    if not category_text:
        return None
    t = category_text.lower().strip()
    if t in CATEGORY_GROUP_RULES:
        return t
    for group in _CATEGORY_GROUP_ORDER:
        if any(kw in t for kw in CATEGORY_GROUP_RULES[group]):
            return group
    return None


def load_products(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["valid_csv_path"])

    df["price_num"] = (
        df["price"] if pd.api.types.is_numeric_dtype(df["price"])
        else df["price"].apply(_parse_price)
    ).fillna(0.0)

    df["orig_num"] = (
        df["original_price"] if pd.api.types.is_numeric_dtype(df["original_price"])
        else df["original_price"].apply(_parse_price)
    ).fillna(0.0)

    df["discount_num"] = (
        df["discount"] if pd.api.types.is_numeric_dtype(df["discount"])
        else df["discount"].apply(_parse_discount)
    ).fillna(0.0)

    df["is_new_bool"] = df["is_new"].apply(_parse_bool)

    if "gender" not in df.columns:
        df["gender"] = df.apply(_infer_gender, axis=1)

    if "category" not in df.columns:
        df["category"] = df.apply(_build_category, axis=1)

    df["category_group"] = df.apply(_infer_category_group, axis=1)

    return df


df_valid = load_products(CONFIG)
print(f"✓ Loaded {len(df_valid)} products")
print(f"✓ Top categories:\n{df_valid['category'].value_counts().head(8)}\n")
print(f"✓ Category groups:\n{df_valid['category_group'].value_counts()}\n")


# ══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING LAYER — CLIP encoder (modern transformers, no fashion-clip pkg)
# ══════════════════════════════════════════════════════════════════════════

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_features(out):
    if torch.is_tensor(out):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        val = getattr(out, attr, None)
        if val is not None:
            return val
    raise TypeError(f"Unexpected output type from CLIP: {type(out)}")


class FashionCLIPCompat:
    def __init__(self, model_name: str = CONFIG["clip_model_name"]):
        self.model = CLIPModel.from_pretrained(model_name).to(_DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_text(self, texts: list, batch_size: int = 1) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True,
            max_length=77,
        ).to(_DEVICE)
        out = self.model.get_text_features(**inputs)
        return _extract_features(out).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 1) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(_DEVICE)
        out = self.model.get_image_features(**inputs)
        return _extract_features(out).cpu().numpy()

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


fclip = FashionCLIPCompat()
print("✓ FashionCLIP (compat) loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE LAYER — Qdrant
# ══════════════════════════════════════════════════════════════════════════

qdrant = QdrantClient(path=LOCAL_QDRANT)
COLLECTION = CONFIG["collection_image"]
MM_COLLECTION = CONFIG["collection_mm"]

_existing_cols = [c.name for c in qdrant.get_collections().collections]
print(f"✓ Collections: {_existing_cols}")

if COLLECTION not in _existing_cols or MM_COLLECTION not in _existing_cols:
    raise RuntimeError(
        f"❌ Expected '{COLLECTION}' and '{MM_COLLECTION}'.\n"
        f"   Found: {_existing_cols}"
    )


def db_info():
    print("\n📦 Qdrant Storage Summary")
    for name in (COLLECTION, MM_COLLECTION):
        try:
            info = qdrant.get_collection(name)
            print(f"   '{name}': {info.points_count} vectors, "
                  f"dim={info.config.params.vectors.size}")
        except Exception:
            print(f"   '{name}': not found")

    print(f"\n   Products  : {len(df_valid)}")
    print(f"   Categories: {df_valid['category'].nunique()}")
    for cat, cnt in df_valid["category"].value_counts().head(5).items():
        print(f"     {cat}: {cnt}")
    print()


def backfill_missing_genders(collection: str = None, batch_size: int = 200,
                              dry_run: bool = False) -> dict:
    collection = collection or MM_COLLECTION
    scanned = 0
    updated = 0
    by_gender = {"women": 0, "men": 0, "kids": 0, "unisex": 0}

    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            scanned += 1
            payload = dict(p.payload or {})
            g = str(payload.get("gender", "")).strip().lower()
            if g in ("women", "men", "kids", "unisex"):
                continue

            new_gender = _row_gender(payload)
            by_gender[new_gender] = by_gender.get(new_gender, 0) + 1
            updated += 1

            if not dry_run:
                qdrant.set_payload(
                    collection_name=collection,
                    payload={"gender": new_gender},
                    points=[p.id],
                )

        if offset is None:
            break

    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}✓ Gender backfill on '{collection}': scanned {scanned}, "
          f"updated {updated} ({by_gender})")
    return {"scanned": scanned, "updated": updated, "by_gender": by_gender}


def ensure_category_group_index(collection: str = None):
    collection = collection or MM_COLLECTION
    try:
        qdrant.create_payload_index(
            collection_name=collection,
            field_name="category_group",
            field_schema="keyword",
        )
        print(f"✓ Created keyword payload index on 'category_group' for '{collection}'")
    except Exception as e:
        print(f"ℹ️ category_group index on '{collection}': {e} (likely already exists)")


def backfill_missing_category_groups(collection: str = None, batch_size: int = 200,
                                      dry_run: bool = False) -> dict:
    collection = collection or MM_COLLECTION
    scanned = 0
    updated = 0
    by_group = {}

    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            scanned += 1
            payload = dict(p.payload or {})
            existing = str(payload.get("category_group", "")).strip().lower()
            if existing and existing != "other":
                continue

            new_group = _infer_category_group(payload)
            by_group[new_group] = by_group.get(new_group, 0) + 1

            if new_group != existing:
                updated += 1
                if not dry_run:
                    qdrant.set_payload(
                        collection_name=collection,
                        payload={"category_group": new_group},
                        points=[p.id],
                    )

        if offset is None:
            break

    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}✓ category_group backfill on '{collection}': scanned {scanned}, "
          f"updated {updated} ({by_group})")
    return {"scanned": scanned, "updated": updated, "by_group": by_group}


# ══════════════════════════════════════════════════════════════════════════
# 5. LLM LAYER — Groq helpers
# ══════════════════════════════════════════════════════════════════════════

def resolve_api_key(cfg: dict = CONFIG) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key != "your_groq_api_key_here":
        return key
    raise ValueError(
        "GROQ_API_KEY is missing or still set to the placeholder value.\n"
        "  1. Copy .env.example to .env (next to app.py) if you haven't.\n"
        "  2. Open .env and replace 'your_groq_api_key_here' with a real key "
        "from https://console.groq.com/keys (starts with 'gsk_').\n"
        "  3. Save the file and re-run: python app.py"
    )


def call_llm(system_prompt: str, user_prompt: str,
             max_tokens: int = None, model: str = None) -> str:
    client = Groq(api_key=resolve_api_key())
    resp = client.chat.completions.create(
        model=model or CONFIG["llm_model"],
        max_tokens=max_tokens or CONFIG["llm_max_tokens"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def describe_image(image: Image.Image) -> str:
    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        client = Groq(api_key=resolve_api_key())
        resp = client.chat.completions.create(
            model=CONFIG["vision_model"],
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": (
                        "Describe this fashion item in 1-2 sentences for a search engine. "
                        "Focus on: garment type, color, fabric if visible, style, pattern, "
                        "and occasion. "
                        "Only if the garment's CUT/SHAPE is unmistakably gendered — a tailored "
                        "blazer, a girl's ruffled dress, a men's dress shirt collar, etc — you "
                        "may add who it's cut for: women's, men's, or kids/children's. "
                        "Do NOT infer this from the model's face, hair, body, or how the item is "
                        "styled/posed — base it on the garment shape alone. Plain oversized tees, "
                        "hoodies, sweatshirts, sweatpants, and similar unisex-cut basics should get "
                        "NO gender label, even though a person of a particular gender is wearing "
                        "one in the photo. When unsure, simply omit any gender wording. "
                        "Example: 'black modest long sleeve maxi dress with floral embroidery, "
                        "formal style, women's'. "
                        "Example (unisex basic): 'oversized red cotton crew-neck t-shirt, casual "
                        "everyday style'. "
                        "Output ONLY the description, nothing else."
                    )},
                ],
            }],
        )
        description = resp.choices[0].message.content.strip()
        print(f"   🖼️  Image described: '{description}'")
        return description
    except Exception as e:
        print(f"   ⚠️ Image description failed ({e}) — using image vector only")
        return ""


_key = resolve_api_key()
print(f"✓ Groq key starts with: {_key[:10]}...")
print("✓ Groq test:", call_llm("You are helpful.", "say hi in 3 words", max_tokens=10))


# ══════════════════════════════════════════════════════════════════════════
# 6. QUERY UNDERSTANDING — keyword fallbacks + LLM intent extraction
# ══════════════════════════════════════════════════════════════════════════

COLORS = [
    "anthracite", "asphalt gray", "baby blue", "beige", "biscuit", "bitter chocolate",
    "black", "blue", "brick red", "brown", "burgundy", "camel", "caramel", "cherry color",
    "cinnamon", "copper color", "coral", "cream", "dark beige", "dark blue", "dark green",
    "dark gray", "dark khaki", "dark lilac", "dark mink", "dark navy blue", "dark pink",
    "dark powder pink", "dark purple", "dark rose", "deep blue", "denim blue", "dusty pink",
    "dusty rose", "ecru", "emerald", "forest green", "fuchsia", "garnet", "gold color",
    "golden color", "granite gray", "gray blue", "gray melange", "green", "grey", "ice",
    "icy blue", "indigo", "ivory", "khaki", "latte", "lavender", "lemon yellow", "leopard",
    "light baby blue", "light beige", "light blue", "light gray", "light green",
    "light mink", "light navy blue", "light yellow", "lilac", "maroon", "metallic",
    "midnight blue", "milky brown", "mink", "mint green", "multi color", "mustard",
    "navy blue", "nile green", "nude", "off white", "olive green", "orange", "peach",
    "petrol", "pink", "pistachio green", "powder pink", "purple", "red", "rose", "salmon",
    "sandstone", "saxe blue", "sea green", "silver color", "sky blue", "smoke color",
    "stone color", "tan", "turquoise", "violet", "white", "yellow",
]

FABRICS = [
    "cotton", "polyester", "viscose", "linen", "silk", "chiffon", "jersey", "satin",
    "velvet", "denim", "wool", "acrylic", "modal", "rayon", "lycra", "spandex", "nylon",
    "elastane",
]

# ── PATCH: style / length / sleeve modifiers ────────────────────────────
# Not a structured payload field in the catalog (unlike color/fabric), so
# these are never hard-filtered — they're folded straight into
# `search_query` so CLIP's semantic vector search does the work (it embeds
# length/silhouette words reasonably well: "short red dress" retrieves
# differently from "long red dress"). Covers requests like "same dress but
# shorter" / "نفس الفستان بس قصير" / "same but with long sleeves".
STYLE_MODIFIERS = {
    "mini":         ["short", "mini", "قصير", "قصيرة"],
    "midi":         ["midi", "knee length", "knee-length", "لحد الركبة"],
    "maxi":         ["long", "maxi", "full length", "floor length", "طويل", "طويلة"],
    "sleeveless":   ["sleeveless", "no sleeve", "بدون كم", "من غير كمام"],
    "short_sleeve": ["short sleeve", "short sleeves", "كم قصير"],
    "long_sleeve":  ["long sleeve", "long sleeves", "كم طويل"],
}


def extract_style_modifier(q: str) -> Optional[str]:
    """Keyword fallback — mirrors extract_color()/extract_fabric(). Returns
    the canonical modifier tag (e.g. 'mini', 'long_sleeve') or None."""
    t = f" {(q or '').lower()} "
    for tag, kws in STYLE_MODIFIERS.items():
        if any(kw in t for kw in kws):
            return tag
    return None


def extract_color(q: str) -> Optional[str]:
    q = q.lower()
    for c in sorted(COLORS, key=len, reverse=True):
        if c in q:
            return c
    # PATCH: Arabic color words. Without this, extract_color() only ever
    # matched English text — so "same dress but in white" correctly boosted
    # the text weight in the vector fusion (see explicit_color_override in
    # chat()), but "عايزة نفس الفستان باللون الابيض" silently didn't, and
    # the image's own (black) color kept dominating retrieval. Checked
    # longest-first so e.g. "أبيض عاجي" doesn't get cut short by a
    # substring of a shorter entry.
    for ar, en in sorted(_ARABIC_COLOR_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        if ar in q:
            return en
    return None


# PATCH: Arabic → catalog-color-vocabulary map used by extract_color().
# Keys are common Arabic color words/spelling variants; values are terms
# already in COLORS (or terms normalize_color() will resolve).
_ARABIC_COLOR_MAP = {
    "ابيض": "white", "أبيض": "white", "بيضاء": "white",
    "اسود": "black", "أسود": "black", "سوداء": "black",
    "احمر": "red", "أحمر": "red", "حمراء": "red",
    "ازرق": "blue", "أزرق": "blue", "زرقاء": "blue",
    "اخضر": "green", "أخضر": "green", "خضراء": "green",
    "اصفر": "yellow", "أصفر": "yellow", "صفراء": "yellow",
    "بني": "brown", "بنية": "brown",
    "رمادي": "gray", "رمادية": "gray",
    "وردي": "pink", "وردية": "pink", "زهري": "pink", "زهرية": "pink",
    "بنفسجي": "purple", "بنفسجية": "purple",
    "برتقالي": "orange", "برتقالية": "orange",
    "كحلي": "navy blue", "كحلية": "navy blue",
    "بيج": "beige",
    "ذهبي": "gold color", "ذهبية": "gold color",
    "فضي": "silver color", "فضية": "silver color",
    "كريمي": "cream",
    "عاجي": "ivory",
    "تركواز": "turquoise",
    "خمري": "burgundy",
    "موف": "lilac",
    "فوشيا": "fuchsia",
    "نبيتي": "burgundy",
    "سماوي": "sky blue",
}


def extract_fabric(q: str) -> Optional[str]:
    q = q.lower()
    for f in FABRICS:
        if f in q:
            return f
    return None


def extract_max_price(q: str) -> Optional[float]:
    q = q.lower()
    m = re.search(r'(?:under|below|max|less than|cheaper than)\s*\$?\s*(\d+\.?\d*)', q)
    if m:
        return float(m.group(1))
    m = re.search(r'\$(\d+\.?\d*)\s*(?:or less|max|budget)', q)
    if m:
        return float(m.group(1))
    return None


def extract_is_new(q: str) -> Optional[bool]:
    if any(w in q.lower() for w in ["new arrival", "new in", "just arrived", "latest"]):
        return True
    return None


def extract_badge(q: str) -> Optional[str]:
    if "24" in q and "hour" in q.lower():
        return "Shipped In 24 hrs"
    return None


def extract_gender(q: str) -> Optional[str]:
    t = f" {(q or '').lower()} "
    if any(w in t for w in [" girl", " girls", " boy", " boys", " kid", " kids",
                              " child", " children"]):
        return "kids"
    if any(w in t for w in [" women", " woman", " ladies", " female", " hijab", " abaya"]):
        return "women"
    if re.search(r"\b(men|man|mens|male)\b", t) or "men's" in t or "man's" in t:
        return "men"
    return None


_COLOR_SYNONYMS = {
    "sage green": "green",
    "sage": "green",
    "hunter green": "dark green",
    "seafoam": "mint green",
    "seafoam green": "mint green",
    "champagne": "gold color",
    "taupe": "beige",
    "charcoal": "dark gray",
    "charcoal gray": "dark gray",
    "wine": "burgundy",
    "wine red": "burgundy",
    "blush": "dusty pink",
    "blush pink": "dusty pink",
    "rust": "brick red",
    "terracotta": "brick red",
    "cobalt": "blue",
    "cobalt blue": "blue",
    "teal": "petrol",
    "plum": "dark purple",
}

_STOPWORDS_IN_COLOR = {"color", "colour", "light", "dark", "deep", "bright"}


def normalize_color(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = raw.strip().lower()

    if t in COLORS:
        return t

    if t in _COLOR_SYNONYMS:
        return _COLOR_SYNONYMS[t]

    for c in sorted(COLORS, key=len, reverse=True):
        if c in t or t in c:
            return c

    t_words = [w for w in t.split() if w not in _STOPWORDS_IN_COLOR]
    for c in sorted(COLORS, key=len, reverse=True):
        c_words = set(c.split())
        if any(w in c_words for w in t_words):
            return c

    print(f"   ⚠️ normalize_color: no vocabulary match for '{raw}', using raw term")
    return t


# ── PATCH: force the requested color into the embedded search text ──────
# Bug this fixes: "same dress but in white" over an uploaded black-dress
# photo was returning black items. Two separate causes:
#
#  1. CLIP embeds a photo's actual color very strongly. Even with the
#     text vector saying "white", fusing it 60/40 (or even 20/80 for
#     vague text) with an image vector of a BLACK dress keeps the ranked
#     candidates dominated by black items — the requested color never
#     gets a real chance to surface in the top-K that reaches Qdrant's
#     query_filter/post-filter stage at all.
#  2. If enhance_query()'s LLM call fails, _intent_fallback()'s
#     search_query falls back to the raw image_description — which, for
#     a black-dress photo, literally contains the word "black" — silently
#     overriding the customer's explicit color request in the text used
#     for retrieval (the `color` field itself was still set correctly to
#     "white", but the search TEXT contradicted it).
#
# Fix: whenever `color` is explicitly known, strip any other catalog
# color word out of search_query and force the requested color to the
# front, so the text embedding itself is unambiguous — regardless of
# which code path produced search_query.
def _strip_known_colors(text: str) -> str:
    t = text
    for c in sorted(COLORS, key=len, reverse=True):
        t = re.sub(re.escape(c), "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _force_color_in_query(search_query: str, color: Optional[str]) -> str:
    if not color:
        return search_query
    stripped = _strip_known_colors(search_query)
    return f"{color} {stripped}".strip()


CATEGORY_KEYWORDS = {
    "shoes":       ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
                     "sandal", "sandals", "heel", "heels"],
    "bags":        ["bag", "bags", "backpack", "handbag", "clutch", "purse", "tote"],
    "dresses":     ["dress", "dresses", "gown", "abaya", "abayas", "jumpsuit"],
    "hijabs":      ["hijab", "hijabs", "scarf", "scarves", "shawl"],
    "tops":        ["shirt", "shirts", "blouse", "blouses", "top", "tops", "tunic",
                     "sweatshirt", "hoodie"],
    "outerwear":   ["jacket", "jackets", "coat", "coats", "cardigan", "blazer"],
    "bottoms":     ["trouser", "trousers", "jean", "jeans", "pant", "pants",
                     "skirt", "skirts", "short", "shorts"],
    "swimwear":    ["swim", "swimwear", "bikini"],
    "accessories": ["accessory", "accessories", "jewelry", "belt", "watch",
                     "sunglasses"],
    "suits":       ["suit", "suits"],
}
_CATEGORY_KEYWORD_ORDER = ["swimwear", "shoes", "bags", "hijabs", "outerwear",
                            "suits", "dresses", "bottoms", "tops", "accessories"]


def extract_category(q: str) -> Optional[str]:
    t = f" {(q or '').lower()} "
    for cat in _CATEGORY_KEYWORD_ORDER:
        if any(kw in t for kw in CATEGORY_KEYWORDS[cat]):
            return cat
    return None


_KIDS_NAME_PATTERN = re.compile(
    r"\bboys?[’'`]?\b|\bgirls?[’'`]?\b|\bkids?\b|\bchild(?:ren)?\b|\bjunior\b|"
    r"\btoddler\b|\bbaby\b|\bbabies\b|\d+\s*-\s*\d+\s*ya[sş]|\d+\s*-\s*\d+\s*(?:yr|year)s?",
    re.IGNORECASE,
)


def _row_gender(row: dict) -> str:
    name = str(row.get("name", ""))
    taxonomy_text = f"{row.get('category', '')} {row.get('articleType', '')} " \
                     f"{row.get('subCategory', '')} {row.get('masterCategory', '')}"

    if _KIDS_NAME_PATTERN.search(name) or _KIDS_NAME_PATTERN.search(taxonomy_text):
        return "kids"

    g = str(row.get("gender", "")).strip().lower()
    if g in ("women", "men", "kids", "unisex"):
        return g

    text = f"{name} {taxonomy_text}".lower()
    if any(w in text for w in ["girl", "boy", "kids", "child"]):
        return "kids"
    if any(w in text for w in ["women", "woman", "hijab", "abaya", "modest dress"]):
        return "women"
    if any(w in text for w in [" men ", " man ", "men's"]):
        return "men"
    return "unisex"


# ── PATCH: added a "style_modifier" field + a rule telling the LLM to
# rewrite search_query for "same but shorter / long sleeves" style edits,
# instead of just parroting back the image's own length/sleeve as-is.
_INTENT_PROMPT_TEMPLATE = """You are a fashion search assistant for Modanisa — a modest fashion e-commerce store.
Analyze the customer input below and extract structured search parameters.
Input may be Arabic, English, or a mix. Handle all cases.

Customer input:
{combined}

Available categories: dresses, abayas, hijabs, scarves, sweatshirts, blouses, tops,
trousers, jeans, skirts, jackets, coats, hoodies, suits, swimwear, shorts, jumpsuits,
tunics, shoes, boots, bags, accessories

Available fabrics: cotton, polyester, viscose, linen, silk, chiffon, jersey, satin,
velvet, denim, wool, acrylic, modal, rayon, lycra, spandex, nylon, elastane

Available colors: {colors}

Rules:
- For "color", pick the CLOSEST matching term from the Available colors list above —
  do not invent color names outside that list (e.g. if the item looks "sage green",
  output "green" or "sea green", whichever fits better; if it looks "charcoal", output
  "dark gray"). If truly nothing fits, output your best free-text guess anyway — a
  downstream normalizer will map it to the closest catalog color.
- If text says "like this but in black" and image shows a blue dress → color = "black"
- If text asks for a length/fit variation relative to the image ("same but shorter",
  "same but with long sleeves", "بس قصير", "نفس الفستان بس كم طويل"), set
  style_modifier to one of: mini, midi, maxi, sleeveless, short_sleeve, long_sleeve —
  and REWRITE search_query to describe the MODIFIED item, not the item exactly as it
  appears in the image (e.g. image = long-sleeve maxi dress, text = "same but short
  sleeves" → search_query should describe a "short sleeve maxi dress...", not repeat
  "long sleeve").
- If text is vague (e.g. "something like this") → extract color/style from image description
- gender must reflect the GARMENT'S CUT/SHAPE, not the person modeling it in the photo.
  Only set gender to women/men/kids if the garment shape itself is unmistakably gendered
  (an evening gown, a tailored men's suit, a girl's ruffled dress, etc). Plain oversized
  tees, hoodies, sweatshirts, sweatpants and similar unisex-cut basics must get gender = null,
  even if a particular gender is visibly wearing the item in the photo. When in doubt, output
  null — do not guess.
- search_query must be in English, 6-12 words, optimized for semantic vector search
- Merge the best signals from both text and image

Respond ONLY with valid JSON — no markdown, no explanation:
{{
  "search_query":      "optimized English search query",
  "color":             "color in English or null",
  "category":          "one category from the list above or null",
  "max_price":         number_or_null,
  "fabric":            "one fabric or null",
  "size":              "e.g. XL, 42, L or null",
  "is_new":            true_or_null,
  "fast_shipping":     true_or_null,
  "discount":          true_or_null,
  "occasion":          "formal, casual, wedding, beach, sport or null",
  "gender":            "women, men, kids or null",
  "style_modifier":    "mini, midi, maxi, sleeveless, short_sleeve, long_sleeve or null",
  "original_language": "arabic or english"
}}"""


def _intent_fallback(raw_query: str, image_description: str) -> dict:
    fallback_q = image_description or raw_query
    return {
        "search_query": fallback_q,
        "color": extract_color(raw_query),
        "category": extract_category(raw_query) or extract_category(image_description),
        "max_price": extract_max_price(raw_query),
        "fabric": extract_fabric(raw_query),
        "size": None,
        "is_new": extract_is_new(raw_query),
        "fast_shipping": None,
        "discount": None,
        "occasion": None,
        "gender": extract_gender(raw_query) or extract_gender(image_description),
        "style_modifier": extract_style_modifier(raw_query),
        "original_language": "english",
    }


def enhance_query(raw_query: str, image_description: str = "") -> dict:
    if image_description and raw_query:
        combined = f'Text query: "{raw_query}"\nImage shows: {image_description}'
    elif image_description:
        combined = f'Image shows: {image_description}'
    else:
        combined = f'Text query: "{raw_query}"'

    prompt = _INTENT_PROMPT_TEMPLATE.format(
        combined=combined,
        colors=", ".join(COLORS),
    )

    try:
        text = call_llm("You are a strict JSON-only fashion search parser.", prompt)
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        print(f"   🧠 Intent: {json.dumps(parsed, ensure_ascii=False)}")
        return parsed
    except Exception as e:
        print(f"   ⚠️ enhance_query failed ({e}) — using fallback")
        return _intent_fallback(raw_query, image_description)


def llm_rerank(query: str, rows: list, top_k: int) -> list:
    if len(rows) <= top_k:
        return rows

    candidates = "\n".join(
        f"{i + 1}. {r.get('name', '')} | {r.get('colors', '')} | "
        f"${r.get('price', '?')} | fabric:{r.get('fabrics', '')} | sizes:{r.get('sizes', '')}"
        for i, r in enumerate(rows)
    )
    prompt = (
        f'You are a fashion search re-ranker.\nQuery: "{query}"\n\n'
        f'Candidates:\n{candidates}\n\n'
        f'Score each candidate 1-5 for relevance (5=perfect). '
        f'Respond ONLY with a JSON array of integers, one per candidate.\n'
        f'Example for 4 candidates: [5, 3, 4, 2]'
    )
    try:
        text = call_llm("You are a strict JSON-only re-ranker.", prompt, max_tokens=100)
        text = text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        if isinstance(scores, list) and len(scores) == len(rows):
            ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
            print(f"   🤖 Re-ranked → top scores: {sorted(scores, reverse=True)[:top_k]}")
            return [r for _, r in ranked[:top_k]]
    except Exception as e:
        print(f"   ⚠️ Re-rank failed ({e}) — using similarity order")
    return rows[:top_k]


# ══════════════════════════════════════════════════════════════════════════
# 7. RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════

CHAT_SYSTEM_PROMPT = """\
You are Nour, a warm personal shopping assistant for Modanisa — a modest fashion platform.
- Base answers ONLY on products in <products> tags.
- Mention 2-3 best matches by name, color, fabric, sizes, price, and discount.
- Highlight 🚚 fast shipping or 🆕 new arrival badges when present.
- For specific colors, only recommend matching products.
- If told a color/size wasn't an exact match, say clearly these are the closest
  available options — never imply they match a request they don't.
- If the customer uploaded an image, acknowledge what you understood from it.
- Reply in Arabic if the customer wrote in Arabic, otherwise reply in English.
- Keep replies to 3-5 warm, conversational sentences. No bullet points."""


def _build_product_lines(results_df: pd.DataFrame, color: Optional[str]) -> list:
    lines = []
    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        name = str(row.get("name", "Unknown"))
        colours = str(row.get("colors", ""))
        fabrics = str(row.get("fabrics", ""))
        sizes = str(row.get("sizes", ""))
        category = str(row.get("category", ""))
        price = f"${row.get('price', 'N/A')}"
        orig = f"${row.get('original_price', 'N/A')}"
        discount = row.get("discount", 0)
        badges = str(row.get("badges", ""))
        is_new = row.get("is_new", False)

        disc_str = f" (-{discount:.0f}%)" if discount else ""
        badge_str = f" | 🚚 {badges}" if badges else ""
        new_str = " | 🆕 New Arrival" if is_new else ""
        match = ""
        if color:
            match = (" ✓ COLOR MATCH"
                      if color.lower() in name.lower() or color.lower() in colours.lower()
                      else " ✗ no color match")

        lines.append(
            f"{rank}. [{category}] {name}\n"
            f"   Colors: {colours} | Fabric: {fabrics} | Sizes: {sizes}\n"
            f"   Price: {price}{disc_str} (was {orig})"
            f"{badge_str}{new_str} | sim: {row.get('similarity', 0):.3f}{match}"
        )
    return lines


def _matches_category(row: dict, variants: set) -> bool:
    for field in ("category", "articleType", "subCategory"):
        val = str(row.get(field, "")).lower()
        if any(v in val for v in variants):
            return True
    return False


def _safe_filter(rows: list, filtered: list, label: str, k: int, verbose: bool) -> list:
    if len(filtered) >= min(2, k):
        if verbose:
            print(f"   ✓ {label}: {len(filtered)} candidates")
        return filtered
    if verbose:
        print(f"   ⚠️ {label} too strict ({len(filtered)}), skipping")
    return rows


# ── PATCH: color moved OUT of _apply_post_filters (now color=None is always
# passed in from chat()) and into apply_color_filter_with_fallback() below,
# so a zero-exact-match color no longer dead-ends the search — it falls
# back to the closest candidates and tells the customer honestly.
def _apply_post_filters(rows: list, *, color, category, fabric, size,
                         want_disc, gender, k: int, verbose: bool) -> list:
    # Gender is enforced strictly and FIRST — it does NOT fall back to the
    # unfiltered set when it narrows things down a lot. Showing a men's item
    # on a women's-item search (or vice versa) is a correctness bug, not a
    # "too strict, loosen up" situation.
    if gender:
        matched = [r for r in rows if _row_gender(r) in (gender, "unisex")]
        if verbose:
            print(f"   ✓ gender '{gender}': {len(matched)}/{len(rows)} candidates kept "
                  f"(cross-gender items excluded, no fallback)")
        rows = matched
    else:
        matched = [r for r in rows if _row_gender(r) != "kids"]
        if verbose and len(matched) != len(rows):
            print(f"   ✓ gender unknown: excluded {len(rows) - len(matched)} kids' "
                  f"item(s) by default ({len(matched)}/{len(rows)} kept)")
        rows = matched

    # NOTE: color is intentionally NOT handled here anymore — see
    # apply_color_filter_with_fallback(), called separately from chat()
    # after this function returns, so it can fall back to "closest match"
    # instead of _safe_filter's blunt keep-everything-or-nothing behavior.
    if color:
        matched = [r for r in rows
                   if color.lower() in str(r.get("colors", "")).lower()
                   or color.lower() in str(r.get("name", "")).lower()]
        if verbose:
            print(f"   ✓ color '{color}': {len(matched)}/{len(rows)} candidates kept "
                  f"(strict, no fallback) — NOTE: chat() should pass color=None here "
                  f"and use apply_color_filter_with_fallback() instead")
        rows = matched

    if category:
        cat_l = category.lower()
        variants = {cat_l, cat_l.rstrip("s"), cat_l + "s"}
        matched = [r for r in rows if _matches_category(r, variants)]
        rows = _safe_filter(rows, matched, f"category '{category}'", k, verbose)

    if fabric:
        matched = [r for r in rows if fabric.lower() in str(r.get("fabrics", "")).lower()]
        rows = _safe_filter(rows, matched, f"fabric '{fabric}'", k, verbose)

    if size:
        matched = [r for r in rows if str(size).lower() in str(r.get("sizes", "")).lower()]
        rows = _safe_filter(rows, matched, f"size '{size}'", k, verbose)

    if want_disc:
        matched = [r for r in rows
                   if float(str(r.get("discount", "0")).replace("%", "").strip() or 0) > 10]
        rows = _safe_filter(rows, matched, "discount>10%", k, verbose)

    return rows


# ── PATCH: closest-match fallback for color ─────────────────────────────
def apply_color_filter_with_fallback(rows: list, color: Optional[str], verbose: bool):
    """Returns (rows, color_matched: bool).

    color_matched=True  → `rows` are exact color matches.
    color_matched=False → no product in `color` exists among the current
    candidates, so `rows` is the closest-match pool (same gender/category/
    price/fabric/size constraints already applied, just not this color).
    chat() threads color_matched into meta + the LLM prompt so the reply
    says plainly these are the nearest options, not an exact hit."""
    if not color:
        return rows, True

    matched = [r for r in rows
               if color.lower() in str(r.get("colors", "")).lower()
               or color.lower() in str(r.get("name", "")).lower()]

    if matched:
        if verbose:
            print(f"   ✓ color '{color}': {len(matched)}/{len(rows)} exact matches")
        return matched, True

    if verbose:
        print(f"   ⚠️ color '{color}': 0 exact matches — falling back to "
              f"closest {len(rows)} candidates (same style/fabric, different color)")
    return rows, False


def _build_query_vector(search_query: str, query_image, alpha: float, verbose: bool,
                         skip_vague_downweight: bool = False):
    text_vec = image_vec = None

    if search_query.strip():
        t = fclip.encode_text([search_query], batch_size=1)
        text_vec = fclip.normalize(t)[0]

    if query_image is not None:
        v = fclip.encode_images([query_image.convert("RGB")], batch_size=1)
        image_vec = fclip.normalize(v)[0]

    if text_vec is not None and image_vec is not None:
        # PATCH: skip_vague_downweight — when chat() has already boosted
        # `alpha` because the customer explicitly typed a color override
        # (e.g. "same dress but in white"), don't let the generic
        # "short query = vague = trust the image more" heuristic below
        # undo that boost. A forced query like "white maxi dress modest
        # style" can still be short, but it's not vague — it's precise.
        is_vague_text = (not skip_vague_downweight) and len(search_query.split()) < 4
        eff_alpha = 0.2 if is_vague_text else alpha
        fused = eff_alpha * text_vec + (1.0 - eff_alpha) * image_vec
        mode = f"Hybrid (α={eff_alpha:.1f})"
    elif text_vec is not None:
        fused, mode = text_vec, "Text only"
    else:
        fused, mode = image_vec, "Image only"

    query_vec = fused / np.linalg.norm(fused)
    if verbose:
        print(f"   🔀 mode         : {mode}")
    return query_vec, mode


@dataclass
class ChatResponse:
    assistant_message: str
    products: list = dc_field(default_factory=list)
    meta: dict = dc_field(default_factory=dict)

    def display(self):
        print("\n" + "─" * 64)
        print(f"🔍 Mode      : {self.meta.get('mode', '?')}")
        print(f"📋 Query used: {self.meta.get('search_query', '?')}")
        if self.meta.get("gender"):
            print(f"🚻 Gender    : {self.meta['gender']}"
                  f"{' (hard filter)' if self.meta.get('gender_is_hard') else ' (soft filter)'}")
        if self.meta.get("category_group"):
            print(f"📦 Category  : {self.meta['category_group']} (hard filter)")
        if self.meta.get("color") and self.meta.get("color_matched") is False:
            print(f"🎨 Color     : {self.meta['color']} requested — NOT matched, "
                  f"showing closest alternatives")
        if self.meta.get("image_description"):
            print(f"🖼️  Image     : {self.meta['image_description']}")
        print("─" * 64 + "\n🤖  Assistant\n")
        print(self.assistant_message)
        if self.products:
            print(f"\n{'─' * 64}\n🛍️  Products ({len(self.products)})\n")
            for i, p in enumerate(self.products, 1):
                disc = f" (-{p.get('discount', 0):.0f}%)" if p.get('discount') else ""
                badge = p.get('badges', '')
                is_new = p.get('is_new', False)
                print(f"  {i}. {p.get('name', 'Unknown')}")
                print(f"     Brand    : {p.get('brand', 'N/A')}")
                print(f"     Gender   : {p.get('gender', 'N/A')}")
                print(f"     Category : {p.get('category', 'N/A')}")
                print(f"     Color    : {p.get('colors', 'N/A')}")
                print(f"     Fabric   : {p.get('fabrics', 'N/A')}")
                print(f"     Sizes    : {p.get('sizes', 'N/A')}")
                print(f"     Price    : ${p.get('price', 'N/A')}{disc}")
                print(f"     Original : ${p.get('original_price', 'N/A')}")
                if badge:
                    print(f"     Badge    : 🚚 {badge}")
                if is_new:
                    print(f"     Status   : 🆕 New Arrival")
                print(f"     Match    : {p.get('similarity', 0):.1%}")
                print(f"     URL      : {p.get('product_url', '')}\n")
        print("─" * 64)

    def show_images(self):
        if not self.products:
            print("No products to display.")
            return
        n = len(self.products)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
        if n == 1:
            axes = [axes]
        for ax, p in zip(axes, self.products):
            img = _load_any(p.get("image_url", ""))
            if img:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#eee")
            disc = f" -{p.get('discount', 0):.0f}%" if p.get('discount') else ""
            badge = "🚚" if p.get("badges") else ""
            is_new = "🆕" if p.get("is_new") else ""
            ax.set_title(
                f"{str(p.get('name', ''))[:26]}\n"
                f"{p.get('colors', '')} | {p.get('fabrics', '')}\n"
                f"${p.get('price', 'N/A')}{disc} {badge}{is_new}\n"
                f"sim={p.get('similarity', 0):.3f}",
                fontsize=6.5,
            )
            ax.axis("off")
        plt.suptitle(f"Results: '{self.meta.get('search_query', '')}'", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def show_all_images(self, product_index: int = 0):
        if product_index >= len(self.products):
            print(f"No product at index {product_index}.")
            return
        p = self.products[product_index]
        urls = [u.strip() for u in str(p.get("all_images", "")).split(",") if u.strip()]
        if not urls:
            print("No extra images available.")
            return
        n = len(urls)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, url in zip(axes, urls):
            img = _load_any(url)
            if img:
                ax.imshow(img)
            else:
                ax.set_facecolor("#eee")
            ax.axis("off")
        plt.suptitle(str(p.get("name", "")), fontsize=9)
        plt.tight_layout()
        plt.show()


_SESSION = None


def _load_any(url: str) -> Optional[Image.Image]:
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        })
    try:
        r = _SESSION.get(str(url).strip(), timeout=CONFIG["img_fetch_timeout"])
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def chat(
    query_text=None,
    query_image=None,
    alpha: float = 0.6,
    k=None,
    category_filter=None,
    gender_filter=None,
    score_threshold=None,
    use_rerank: bool = False,
    verbose: bool = True,
    max_price_override: Optional[float] = None,
) -> ChatResponse:

    k = k or CONFIG["rag_top_k"]
    score_threshold = score_threshold or CONFIG["score_threshold"]
    has_text = bool(query_text and query_text.strip())
    has_image = query_image is not None

    if not has_text and not has_image:
        return ChatResponse("Please provide a text query or upload an image.", [], {})

    if verbose:
        print(f"\n{'=' * 64}\n🛍️  Input: text={bool(has_text)} | image={has_image}")

    # STEP 1 — describe image
    image_description = describe_image(query_image) if has_image else ""

    # STEP 2 — LLM intent extraction
    intent = enhance_query(query_text or "", image_description)
    search_query = intent.get("search_query") or query_text or image_description or ""

    color = intent.get("color") or extract_color(query_text or "")
    color = normalize_color(color)
    fabric = intent.get("fabric") or extract_fabric(query_text or "")

    # PATCH: style/length modifier — "same dress but shorter" etc. Folded
    # into search_query (semantic signal for CLIP) rather than a structured
    # filter, since there's no such payload field in the catalog.
    style_modifier = intent.get("style_modifier") or extract_style_modifier(query_text or "")

    # ── Gender resolution: EXPLICIT vs INFERRED ─────────────────────────
    gender_explicit = gender_filter or extract_gender(query_text or "")
    gender_inferred = intent.get("gender") or extract_gender(image_description)
    gender_hard = gender_explicit
    gender_soft = gender_explicit or gender_inferred

    max_price = max_price_override or intent.get("max_price") or extract_max_price(query_text or "")
    is_new = intent.get("is_new") or extract_is_new(query_text or "")
    badge = "Shipped In 24 hrs" if intent.get("fast_shipping") else extract_badge(query_text or "")
    want_disc = bool(intent.get("discount"))
    size = intent.get("size")
    occasion = intent.get("occasion")

    category = (
        category_filter
        or intent.get("category")
        or extract_category(query_text or "")
        or extract_category(image_description)
    )
    category_group = _resolve_category_group(category)

    orig_lang = intent.get("original_language", "english")

    if occasion and occasion.lower() not in search_query.lower():
        search_query = f"{search_query} {occasion}"

    # PATCH: merge style_modifier into search_query (same pattern as occasion)
    if style_modifier and style_modifier.replace("_", " ") not in search_query.lower():
        search_query = f"{search_query} {style_modifier.replace('_', ' ')}"

    # PATCH: force the requested color into the embedded search text and,
    # when the customer typed the color themselves (vs. it only being
    # inferred from the photo), lean the fused vector much more heavily
    # toward text. Fixes "same dress but in white" over a black-dress
    # photo returning black items — see _force_color_in_query()'s
    # docstring above for the full root cause.
    explicit_color_override = has_image and bool(extract_color(query_text or ""))
    if color:
        search_query = _force_color_in_query(search_query, color)

    effective_alpha = 0.85 if explicit_color_override else alpha

    if verbose:
        print(f"   📋 search_query    : '{search_query}'")
        print(f"   🎨 color           : {color}"
              f"{' (explicit text override → alpha boosted to 0.85)' if explicit_color_override else ''}")
        print(f"   🧵 fabric          : {fabric}")
        print(f"   📏 style_modifier  : {style_modifier}")
        print(f"   🚻 gender          : hard={gender_hard} soft={gender_soft}")
        print(f"   💰 max_price       : {max_price}")
        print(f"   📦 category        : {category}")
        print(f"   📦 category_group  : {category_group} "
              f"{'(hard filter)' if category_group else '(no hard filter — unresolved)'}")
        print(f"   📐 size            : {size}")

    # STEP 3 — fused query vector
    query_vec, mode = _build_query_vector(
        search_query, query_image, effective_alpha, verbose,
        skip_vague_downweight=explicit_color_override,
    )

    # STEP 4 — Qdrant vector search
    must = []
    if max_price:
        must.append(FieldCondition(key="price", range=Range(lte=float(max_price))))
    if is_new:
        must.append(FieldCondition(key="is_new", match=MatchValue(value=True)))
    if badge:
        must.append(FieldCondition(key="badges", match=MatchValue(value=badge)))
    if gender_hard:
        must.append(FieldCondition(key="gender", match=MatchAny(any=[gender_hard, "unisex"])))
    if category_group:
        must.append(FieldCondition(key="category_group", match=MatchValue(value=category_group)))

    # PATCH: pull a deeper candidate pool when there's an explicit color
    # request over an image search — gives apply_color_filter_with_fallback()
    # an actual chance to find real color matches instead of only seeing
    # the (visually similar-to-the-photo, i.e. same-color) top few hits.
    # PATCH: pull a bit more headroom than before — some candidates will now
    # get dropped by the no-image filter right after this query, so fetching
    # only exactly k*fetch_multiplier risked ending up with fewer than k
    # results even when enough real (image-having) matches existed.
    fetch_k = max(k * CONFIG["fetch_multiplier"], 20)
    if color:
        fetch_k = max(fetch_k, k * 6)
    response = qdrant.query_points(
        collection_name=MM_COLLECTION,
        query=query_vec.tolist(),
        limit=fetch_k,
        query_filter=Filter(must=must) if must else None,
        with_payload=True,
        with_vectors=False,
    )
    rows = [{**dict(hit.payload), "similarity": round(hit.score, 4)} for hit in response.points]

    # PATCH: drop candidates with no usable image before anything else, so
    # the customer never sees a blank "No image available" card. A product
    # counts as "has an image" if image_url is a non-empty, non-"nan" string.
    def _has_image(row: dict) -> bool:
        url = str(row.get("image_url", "")).strip()
        return bool(url) and url.lower() not in ("nan", "none")

    before_img_filter = len(rows)
    rows = [r for r in rows if _has_image(r)]
    if verbose and len(rows) != before_img_filter:
        print(f"   🖼️  image filter: dropped {before_img_filter - len(rows)} item(s) "
              f"with no image_url ({len(rows)} remaining)")

    if not rows:
        msg = "Couldn't find any products. Try different keywords or a different image."
        reasons = []
        if gender_hard:
            source = "the Gender filter dropdown" if gender_filter else "your search text"
            reasons.append(f"gender locked to '{gender_hard}' (from {source})")
        if category_group:
            reasons.append(f"category locked to '{category_group}'")
        if reasons:
            msg = (
                f"Couldn't find any matches ({', '.join(reasons)}). "
                f"Try loosening a filter, or use different keywords / a different image."
            )
        return ChatResponse(
            msg, [], {"mode": mode, "search_query": search_query,
                      "gender": gender_hard, "gender_is_hard": bool(gender_hard),
                      "category_group": category_group},
        )

    # STEP 5 — client-side post-filters. PATCH: color=None here on purpose —
    # color is now handled separately below via
    # apply_color_filter_with_fallback() so it can degrade to "closest
    # match" instead of a dead end.
    rows = _apply_post_filters(
        rows, color=None, category=category, fabric=fabric, size=size,
        want_disc=want_disc, gender=gender_soft, k=k, verbose=verbose,
    )
    if not rows:
        return ChatResponse(
            "Found products but none were close enough to your query. Try being more specific.",
            [], {"mode": mode, "search_query": search_query,
                 "gender": gender_soft, "gender_is_hard": bool(gender_hard),
                 "category_group": category_group},
        )

    # PATCH: color fallback — degrade to closest-match instead of dead end
    rows, color_matched = apply_color_filter_with_fallback(rows, color, verbose)
    if not rows:
        return ChatResponse(
            "Found products but none were close enough to your query. Try being more specific.",
            [], {"mode": mode, "search_query": search_query,
                 "gender": gender_soft, "gender_is_hard": bool(gender_hard),
                 "category_group": category_group, "color": color, "color_matched": color_matched},
        )

    rows = [r for r in rows if r["similarity"] >= score_threshold]
    if not rows:
        msg = "Found products but none were close enough to your query. Try being more specific."
        return ChatResponse(
            msg, [], {"mode": mode, "search_query": search_query,
                      "gender": gender_soft, "gender_is_hard": bool(gender_hard),
                      "category_group": category_group},
        )

    # STEP 6 — optional LLM re-ranking (still similarity-first: rows already
    # arrive sorted by Qdrant similarity descending; this just refines the
    # order among the top candidates when enabled)
    rows = llm_rerank(search_query, rows, top_k=k) if (use_rerank and len(rows) > k) else rows[:k]
    results_df = pd.DataFrame(rows).reset_index(drop=True)

    # STEP 7 — LLM answer
    lines = _build_product_lines(results_df, color if color_matched else None)
    color_note = ""
    if color and color_matched:
        color_note = f"\n⚠️ Customer wants {color.upper()}. ONLY recommend ✓ COLOR MATCH items.\n"
    elif color and not color_matched:
        # PATCH: strengthened — the previous wording made the LLM stop at
        # a short apology instead of still recommending products. Now it's
        # told explicitly to do BOTH: state plainly it's not in that color,
        # AND still give the normal 2-3 product recommendation (name,
        # actual color, fabric, price) same as any other reply.
        color_note = (
            f"\n⚠️ We do NOT carry this item in {color.upper()}. The products below are the "
            f"closest matches in the same style/fabric/price range, just in different colors.\n"
            f"In your reply: (1) say clearly, in one short sentence, that {color} isn't "
            f"available for this item — (2) then STILL recommend 2-3 of the products below "
            f"by name, their ACTUAL color, fabric, sizes, and price, exactly as you would for "
            f"any normal match — (3) end by asking if they'd like to see other colors. "
            f"Don't stop after the apology — the product recommendation is still required.\n"
        )
    lang_note = "\n🌐 Customer wrote in Arabic — reply in Arabic.\n" if orig_lang == "arabic" else ""
    img_note = (
        f"\n📸 Customer uploaded an image showing: {image_description}\n"
        f"Acknowledge this in your response.\n"
    ) if image_description else ""

    user_prompt = (
        "<products>\n" + "\n".join(lines) + "\n</products>\n"
        + color_note + lang_note + img_note
        + f'\nCustomer query: "{query_text or "(image search)"}"\n\n'
        + "Give a warm, conversational 3-5 sentence recommendation."
    )
    answer = call_llm(CHAT_SYSTEM_PROMPT, user_prompt)
    products = [row.to_dict() for _, row in results_df.iterrows()]

    meta = {
        "mode": mode,
        "search_query": search_query,
        "image_description": image_description,
        "color": color,
        "color_matched": color_matched,       # PATCH — False means fallback/closest-match
        "fabric": fabric,
        "style_modifier": style_modifier,     # PATCH
        "gender": gender_soft,
        "gender_is_hard": bool(gender_hard),
        "category": category,
        "category_group": category_group,
        "n_results": len(products),
        "language": orig_lang,
    }
    return ChatResponse(answer, products, meta)


db_info()


# ══════════════════════════════════════════════════════════════════════════
# 8. PRESENTATION LAYER — Gradio product cards
# ══════════════════════════════════════════════════════════════════════════
#
# ── COLOR PALETTE (only thing changed in sections 8 & 9) ───────────────────
# Cream / camel-brown palette lifted from the reference screenshot. All
# hex values live here so the whole demo's look can be re-themed by editing
# this one dict — nothing else in the pipeline (sections 1-7, 10) changed.
PALETTE = {
    "bg":            "#FBF3EA",  # page background (warm cream)
    "panel":         "#FFFFFF",  # card / panel background
    "panel_alt":     "#FDF8F2",  # soft inner panel (e.g. image dropzone)
    "border":        "#ECDFCD",  # light beige border
    "text":          "#2E2117",  # primary text
    "text_muted":    "#8A7963",  # secondary / muted text
    "accent":        "#A6531E",  # camel / burnt-orange brown (links, accents)
    "accent_dark":   "#7C3D14",  # darker brown (primary buttons)
    "accent_soft":   "#F4E3D3",  # soft tag background
    "discount":      "#D8402F",  # red discount badge
    "new_badge":     "#8E5A2B",  # brown "new arrival" badge (was purple)
    "shipping":      "#4C7A4A",  # muted green fast-shipping badge
}

CUSTOM_CSS = f"""
.gradio-container {{
    background: {PALETTE['bg']} !important;
    color: {PALETTE['text']};
}}
#header-title {{ color: {PALETTE['text']} !important; }}
#header-sub {{ color: {PALETTE['text_muted']} !important; }}
#find-btn, .gr-button-primary {{
    background: {PALETTE['accent_dark']} !important;
    border: none !important;
    color: #fff !important;
}}
#find-btn:hover, .gr-button-primary:hover {{
    background: {PALETTE['accent']} !important;
}}
textarea, input[type=text] {{
    background: {PALETTE['panel_alt']} !important;
    border: 1px solid {PALETTE['border']} !important;
    color: {PALETTE['text']} !important;
}}
"""


def _product_card(row: dict) -> str:
    name = str(row.get("name", "Unknown"))
    colours = str(row.get("colors", ""))
    fabrics = str(row.get("fabrics", ""))
    price = row.get("price", "N/A")
    orig = row.get("original_price", "N/A")
    discount = row.get("discount", 0)
    sizes = str(row.get("sizes", ""))
    brand = str(row.get("brand", ""))
    badges = str(row.get("badges", ""))
    is_new = row.get("is_new", False)
    sim = row.get("similarity", 0)
    image_url = str(row.get("image_url", ""))
    all_imgs = str(row.get("all_images", ""))
    prod_url = str(row.get("product_url", "#"))

    disc_html = (f'<span style="color:{PALETTE["discount"]};font-size:12px;font-weight:600">'
                 f'-{discount:.0f}% OFF</span> ') if discount else ""
    orig_html = (f'<span style="color:#B7A995;font-size:11px;text-decoration:line-through">'
                 f'${orig}</span> ') if orig and orig != "N/A" and discount else ""
    badge_html = (f'<div style="font-size:11px;color:{PALETTE["shipping"]};font-weight:600">'
                  f'🚚 {badges}</div>') if badges and badges not in ("nan", "") else ""
    new_html = (f'<div style="font-size:11px;color:{PALETTE["new_badge"]};font-weight:600">'
                '🆕 New Arrival</div>') if is_new else ""
    fabric_html = (f'<div style="font-size:11px;color:{PALETTE["text_muted"]}">Fabric: {fabrics}</div>'
                   ) if fabrics and fabrics not in ("nan", "") else ""

    img_urls = [u.strip() for u in all_imgs.split(",") if u.strip()][:4]
    thumbs = "".join(
        f'<img src="{u}" referrerpolicy="no-referrer" loading="lazy" '
        f'style="width:36px;height:36px;object-fit:cover;'
        f'border-radius:4px;border:1px solid {PALETTE["border"]}" '
        f'onerror="this.style.display=\'none\'">'
        for u in img_urls[1:]
    )
    thumbs_html = f'<div style="display:flex;gap:4px;margin-top:6px">{thumbs}</div>' if thumbs else ""

    seen = set()
    candidates = []
    for u in [image_url, *img_urls]:
        u = (u or "").strip()
        if u and u.lower() not in ("nan", "none") and u not in seen:
            seen.add(u)
            candidates.append(u)

    placeholder_style = (
        f"width:100%;height:210px;background:{PALETTE['panel_alt']};display:flex;"
        f"align-items:center;justify-content:center;color:{PALETTE['text_muted']};"
        "font-size:12px;text-align:center;padding:0 10px"
    )
    if candidates:
        first, rest = candidates[0], candidates[1:]
        fallback_attr = "|".join(rest)
        img_html = (
            f'<div style="position:relative">'
            f'<img src="{first}" data-fallback="{fallback_attr}" '
            f'referrerpolicy="no-referrer" loading="lazy" '
            f'style="width:100%;height:210px;object-fit:cover;display:block" '
            f'onerror="'
            f"var l=(this.getAttribute('data-fallback')||'').split('|').filter(Boolean);"
            f"if(l.length){{var n=l.shift();this.setAttribute('data-fallback',l.join('|'));this.src=n;}}"
            f"else{{this.style.display='none';this.nextElementSibling.style.display='flex';}}"
            f'">'
            f'<div style="{placeholder_style};display:none">No image available</div>'
            f'</div>'
        )
    else:
        img_html = f'<div style="{placeholder_style}">No image available</div>'

    return f"""
<div style="border:1px solid {PALETTE['border']};border-radius:14px;overflow:hidden;
            background:{PALETTE['panel']};display:flex;flex-direction:column;min-width:170px;
            box-shadow:0 1px 3px rgba(46,33,23,0.05)">
  {img_html}
  <div style="padding:10px;flex:1;display:flex;flex-direction:column;gap:3px">
    <div style="font-size:13px;font-weight:700;line-height:1.3;color:{PALETTE['text']}">{textwrap.shorten(name, 42)}</div>
    <div style="font-size:12px;color:{PALETTE['text_muted']}">{brand}</div>
    <div style="font-size:12px;color:{PALETTE['text_muted']}">{colours}</div>
    {fabric_html}
    <div style="font-size:11px;color:{PALETTE['text_muted']}">Sizes: {textwrap.shorten(sizes, 30)}</div>
    <div style="margin-top:4px">{disc_html}{orig_html}
      <span style="font-size:14px;font-weight:800;color:{PALETTE['text']}">${price}</span>
    </div>
    {badge_html}{new_html}
    <div style="font-size:11px;color:{PALETTE['text_muted']}">Match: {sim:.0%}</div>
    {thumbs_html}
    <a href="{prod_url}" target="_blank"
       style="margin-top:6px;font-size:12px;font-weight:600;color:{PALETTE['accent']};text-decoration:none">
      View on Modanisa →
    </a>
  </div>
</div>"""


def build_products_html(products: list) -> str:
    if not products:
        return f"<p style='color:{PALETTE['text_muted']};padding:12px'>No products found.</p>"
    cards = "".join(_product_card(p) for p in products)
    fallback_script = """
<script>
function modanisaImgFallback(img) {
  var list = (img.getAttribute('data-fallback') || '').split('|').filter(Boolean);
  if (list.length) {
    var next = list.shift();
    img.setAttribute('data-fallback', list.join('|'));
    img.src = next;
  } else {
    img.style.display = 'none';
    if (img.nextElementSibling) { img.nextElementSibling.style.display = 'flex'; }
  }
}
</script>
"""
    return (
        fallback_script
        + '<div style="display:grid;grid-template-columns:'
        'repeat(auto-fill,minmax(170px,1fr));gap:12px;padding:4px">'
        f'{cards}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════
# 9. GRADIO UI
# ══════════════════════════════════════════════════════════════════════════

import gradio as gr

CATEGORIES_UI = [
    "All categories",
    "modest-clothing", "dresses-evening-gowns-z1-en-en", "shoes-bag",
    "sea", "hijab", "outerwear", "plus-size-clothing", "flash-deals-12",
    "dresses", "sweatshirts", "shirts", "blouses", "tops", "trousers", "jeans",
    "skirts", "abayas", "hijabs", "scarves", "jackets", "coats", "cardigans",
    "sweaters", "hoodies", "suits", "swimwear", "shorts", "jumpsuits", "tunics",
    "shoes", "sneakers", "boots", "bags", "accessories", "other",
]
FABRICS_UI = [
    "All fabrics", "cotton", "polyester", "viscose", "linen", "silk", "chiffon",
    "jersey", "satin", "velvet", "denim", "wool", "acrylic", "modal", "rayon",
    "lycra", "spandex", "nylon", "elastane",
]
GENDERS_UI = ["Auto-detect", "women", "men", "kids"]


def demo_search(query_text, query_image, k, category, gender_choice,
                 max_price_str, fabric, show_new_only, fast_shipping, use_rerank_cb):

    has_text = bool(query_text and query_text.strip())
    has_image = query_image is not None
    if not has_text and not has_image:
        return "⚠️ Please enter a text query or upload an image.", "", ""

    max_price_val = None
    if max_price_str and max_price_str.strip():
        try:
            max_price_val = float(max_price_str.replace("$", "").strip())
        except ValueError:
            pass

    extra = ""
    if fabric and fabric != "All fabrics":
        extra += f" {fabric} fabric"
    if show_new_only:
        extra += " new arrival"
    if fast_shipping:
        extra += " shipped in 24 hours"

    augmented = (query_text or "") + extra
    cat_filter = None if (not category or category == "All categories") else category
    gender_override = None if (not gender_choice or gender_choice == "Auto-detect") else gender_choice

    try:
        r = chat(
            query_text=augmented if (has_text or extra) else None,
            query_image=query_image,
            k=int(k),
            category_filter=cat_filter,
            gender_filter=gender_override,
            use_rerank=use_rerank_cb,
            verbose=True,
            max_price_override=max_price_val,
        )
    except Exception as e:
        import traceback
        return f"⚠️ Error: {e}\n{traceback.format_exc()}", "", ""

    mode = r.meta.get("mode", "?")
    n = r.meta.get("n_results", 0)
    sq = r.meta.get("search_query", "")
    img_d = r.meta.get("image_description", "")
    gdr = r.meta.get("gender", "")
    gdr_hard = r.meta.get("gender_is_hard", False)
    cat_grp = r.meta.get("category_group", "")
    color = r.meta.get("color", "")
    color_matched = r.meta.get("color_matched", True)

    status_parts = [f"**Mode:** {mode}", f"**{n} results**", f"Query: _{sq}_"]
    if gdr:
        gender_label = f"Gender: {gdr} (🔒 locked, explicit)" if gdr_hard else f"Gender: {gdr} (soft, inferred)"
        status_parts.append(gender_label)
    else:
        status_parts.append("Gender: Auto-detect")
    if cat_grp:
        status_parts.append(f"Category: {cat_grp} (🔒 locked)")
    if color:
        # PATCH — surface the closest-match flag in the UI status line too
        color_label = f"Color: {color} ✓" if color_matched else f"Color: {color} ✗ (showing closest match)"
        status_parts.append(color_label)
    if img_d:
        status_parts.append(f"Image: _{textwrap.shorten(img_d, 60)}_")
    if cat_filter and cat_filter != cat_grp:
        status_parts.append(f"Category (raw): {cat_filter}")

    return " · ".join(status_parts), r.assistant_message, build_products_html(r.products)


with gr.Blocks(title="Stylo") as demo:
    gr.Markdown(
        "## 🛍️ Modanisa Fashion RAG — Search Assistant",
        elem_id="header-title",
    )
    gr.Markdown(
        "Type a query in **Arabic or English**, and/or upload a **photo** of "
        "an item. The assistant understands the item (from the photo and/or "
        "text), retrieves matching products, and replies in the same "
        "language you used.",
        elem_id="header-sub",
    )
    with gr.Row():
        with gr.Column(scale=1):
            query_text = gr.Textbox(
                label="Search query (Arabic or English)",
                placeholder="e.g. فستان أسود طويل بأكمام  /  long black dress with sleeves",
                lines=2,
            )
            query_image = gr.Image(type="pil", label="Upload a photo (optional)")

            with gr.Row():
                category = gr.Dropdown(CATEGORIES_UI, value="All categories", label="Category")
                gender_choice = gr.Dropdown(GENDERS_UI, value="Auto-detect", label="Gender")
            with gr.Row():
                fabric = gr.Dropdown(FABRICS_UI, value="All fabrics", label="Fabric")
                max_price_str = gr.Textbox(label="Max price", placeholder="e.g. 50")
            with gr.Row():
                k = gr.Slider(1, 30, value=10, step=1, label="Number of results")
            with gr.Row():
                show_new_only = gr.Checkbox(label="New arrivals only")
                fast_shipping = gr.Checkbox(label="Fast shipping (24h)")
                use_rerank_cb = gr.Checkbox(label="Use rerank")

            search_btn = gr.Button("🔍 Search", variant="primary", elem_id="find-btn")

        with gr.Column(scale=2):
            status = gr.Markdown()
            assistant_msg = gr.Markdown()
            products_html = gr.HTML()

    search_btn.click(
        fn=demo_search,
        inputs=[query_text, query_image, k, category, gender_choice,
                max_price_str, fabric, show_new_only, fast_shipping, use_rerank_cb],
        outputs=[status, assistant_msg, products_html],
    )
    query_text.submit(
        fn=demo_search,
        inputs=[query_text, query_image, k, category, gender_choice,
                max_price_str, fabric, show_new_only, fast_shipping, use_rerank_cb],
        outputs=[status, assistant_msg, products_html],
    )


from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = gr.mount_gradio_app(app, demo, path="/gradio")

class SearchRequest(BaseModel):
    query: Optional[str] = None
    k: int = 10
    category: Optional[str] = None
    gender: Optional[str] = None
    max_price: Optional[float] = None
    use_rerank: bool = False


def _clean_optional_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    if not v or v.lower() == "string":
        return None
    return v


# ══════════════════════════════════════════════════════════════════════════
# 10. IMAGE PROXY — fixes Modanisa's CDN hotlink/referer protection blocking
#     <img> requests from a separate frontend (e.g. v0, localhost:3000).
#     The backend fetches the image server-to-server (Modanisa accepts that,
#     since there's no browser Referer/Origin check on that path) and
#     streams the bytes back from this backend's own origin.
# ══════════════════════════════════════════════════════════════════════════

_IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.modanisa.com/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


@app.get("/image-proxy")
async def image_proxy(url: str):
    """GET /image-proxy?url=<url-encoded original Modanisa image URL>
    Fetches the image server-to-server and streams it back so a browser
    on a different origin can load it without hitting Modanisa's hotlink
    protection directly."""
    if not url.startswith("https://"):
        return Response(status_code=400, content="Invalid url")

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=_IMG_HEADERS)
            resp.raise_for_status()
    except Exception as e:
        return Response(status_code=502, content=f"Upstream fetch failed: {e}")

    content_type = resp.headers.get("content-type", "image/jpeg")
    return Response(
        content=resp.content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _proxify(products: list, base_url: str) -> list:
    """Rewrites image_url / all_images on each product dict to point at
    this backend's /image-proxy instead of the raw Modanisa CDN URL."""
    for p in products:
        if p.get("image_url"):
            p["image_url"] = f"{base_url}/image-proxy?url={quote(p['image_url'], safe='')}"
        if p.get("all_images"):
            imgs = [u.strip() for u in str(p["all_images"]).split(",") if u.strip()]
            p["all_images"] = ",".join(
                f"{base_url}/image-proxy?url={quote(u, safe='')}" for u in imgs
            )
    return products


@app.post("/search")
def search(req: SearchRequest, request: Request):
    result = chat(
        query_text=req.query,
        query_image=None,
        k=req.k,
        category_filter=_clean_optional_str(req.category),
        gender_filter=_clean_optional_str(req.gender),
        max_price_override=req.max_price,
        use_rerank=req.use_rerank,
        verbose=False,
    )
    result.products = _proxify(result.products, str(request.base_url).rstrip("/"))
    return {
        "message": result.assistant_message,
        "products": result.products,
        "meta": result.meta,
    }


@app.post("/search-image")
async def search_image(
    request: Request,
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
    k: int = Form(10),
    category: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    max_price: Optional[float] = Form(None),
    use_rerank: bool = Form(False),
):
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return {"error": f"Could not read uploaded image: {e}"}

    result = chat(
        query_text=_clean_optional_str(query),
        query_image=pil_image,
        k=k,
        category_filter=_clean_optional_str(category),
        gender_filter=_clean_optional_str(gender),
        max_price_override=max_price,
        use_rerank=use_rerank,
        verbose=False,
    )
    result.products = _proxify(result.products, str(request.base_url).rstrip("/"))
    return {
        "message": result.assistant_message,
        "products": result.products,
        "meta": result.meta,
    }


@app.get("/health")
def health():
    return {"status": "ok", "products_loaded": len(df_valid)}


if __name__ == "__main__":
    print("🚀 Starting RAG API at http://localhost:5000 (docs: /docs, health: /health)")
    print("🖼️  Gradio UI available at http://localhost:5000/gradio")
    print("🔁 Image proxy available at http://localhost:5000/image-proxy?url=...")
    uvicorn.run(app, host="0.0.0.0", port=5000)