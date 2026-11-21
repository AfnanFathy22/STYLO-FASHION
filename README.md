
# Modanisa Fashion RAG — Local Version

This is your Colab notebook's final pipeline cell, converted to run as a plain
local Python service (no Google Drive mount, no Colab secrets, no ngrok).

## What changed vs. the Colab notebook
- `google.colab.drive.mount(...)` and `google.colab.userdata` are gone.
- All paths now live under a local `data/` folder next to `app.py`, instead of
  `/content/drive/MyDrive/...`.
- The "restore Qdrant from Drive" step is gone — you already have the Qdrant
  folder locally, so the script just points at it directly.
- The Groq API key is read from a local `.env` file instead of a Colab secret.
- The `!pip install` cell and the ngrok tunnel are removed — the FastAPI
  server just runs directly on `http://localhost:8000`.

## 1. Folder layout (placed alongside your IDM-VTON project)

Drop this whole `rag_local` folder into your `Fashion_Project` directory, next
to `IDM-VTON`, so you end up with something like:

```
Fashion_Project/
├── IDM-VTON/              ← your existing virtual try-on code
└── rag_local/              ← this folder
    ├── app.py
    ├── requirements.txt
    ├── .env                ← you create this (see step 3)
    └── data/
        ├── modanisa_valid.csv
        └── qdrant_storage/  ← the Qdrant folder you downloaded from Drive
```

Copy your **CSV file** and your **`qdrant_storage` folder** (the one you
downloaded from Google Drive) into `rag_local/data/` exactly as shown above.
If you'd rather keep them somewhere else, just edit the two paths at the top
of `app.py` (`CONFIG["valid_csv_path"]` and `CONFIG["qdrant_local"]`).

## 2. Create a virtual environment and install dependencies

It's fine to reuse the same `venv` your IDM-VTON project already has in
`Fashion_Project/IDM-VTON/venv`, or make a separate one for this service —
either works, they don't need to share an environment. To make a separate one:

```powershell
cd Fashion_Project\rag_local
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA-accelerated CLIP embedding, install
the CUDA build of torch first (see https://pytorch.org/get-started/locally/),
then `pip install -r requirements.txt` for the rest.

## 3. Set your Groq API key

```powershell
copy .env.example .env
```

Open `.env` and paste your real key in place of `your_groq_api_key_here`.

## 4. Run it

```powershell
python app.py
```

You should see:
```
✅ Using local Qdrant storage at ...
✓ Loaded N products
✓ FashionCLIP (compat) loaded
✓ Collections: ['modanisa_image', 'modanisa_multimodal']
🚀 Starting RAG API at http://localhost:8000  (docs: /docs, health: /health)
🖼️  Gradio UI available at http://localhost:8000/gradio
```

## Using the Gradio UI (easiest way to test)

Open **http://localhost:8000/gradio** in your browser. This is a proper
point-and-click interface for the same "image + Arabic/English query →
understand item → retrieve → recommend" flow:

- Type a query in Arabic, English, or a mix (or leave it blank)
- Optionally upload a photo of an item
- Filter by category / gender / fabric / max price, toggle new-arrivals-only
  or fast-shipping, adjust number of results
- Click **🔍 Search** — see the parsed query, applied filters, the
  assistant's reply (in whichever language you used), and the matched
  products as cards

This is served from the same process as the JSON API — no separate command
needed, `python app.py` starts both.

Check it's alive: open http://localhost:8000/health in a browser, or:
```powershell
curl http://localhost:8000/health
```

Try a search:
```powershell
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d "{\"query\": \"black abaya\", \"k\": 5}"
```

### Image + Arabic/English search (`/search-image`)

This is the full "image + text → understand item → retrieve → recommend"
flow: send an image file and (optionally) a text query in Arabic, English,
or a mix. `chat()` runs the vision model on the image, the LLM on the text
(detecting the language and parsing intent), fuses both into one CLIP
search, and replies back in the same language the customer used.

Test it via the interactive docs (easiest): open
`http://localhost:8000/docs`, expand **POST /search-image**, click
"Try it out", pick an image file, optionally type a query (e.g.
`فستان أسود` or `black dress`), and hit Execute.

Or via curl:
```powershell
curl -X POST http://localhost:8000/search-image ^
  -F "image=@C:\path\to\photo.jpg" ^
  -F "query=فستان أسود للمناسبات" ^
  -F "k=5"
```

The response includes `message` (the assistant's reply, in Arabic or
English to match the query), `products` (the matched items), and `meta`.

## Connecting it to the virtual try-on side

Both this RAG service and your IDM-VTON Gradio app can run at the same time
on your machine, on different ports (RAG on `8000`, Gradio's `app.py` on
whatever port it defaults to, e.g. `7860`). A frontend (or your IDM-VTON UI
code) can call `POST http://localhost:8000/search` to fetch product results,
then hand the chosen product's garment image to the try-on pipeline.

## Notes
- First run will download the `patrickjohncyh/fashion-clip` weights from
  Hugging Face (a few hundred MB) — this needs internet access once, then
  it's cached locally (`~/.cache/huggingface`).
- If Qdrant complains about a lock file from a previous session, `app.py`
  already clears `data/qdrant_storage/.lock` automatically on startup.
