"""
Quick diagnostic: prints every unique hostname found in image_url and
all_images across your CSV, so you know exactly what to put in
CONFIG["image_proxy_allowed_domains"].

Run it from the same folder as app.py (same data/ layout):
    python check_domains.py
"""
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "modanisa_valid.csv"   # adjust if different

df = pd.read_csv(CSV_PATH)

hosts = Counter()

def collect(url):
    if not isinstance(url, str) or not url.strip():
        return
    for u in url.split(","):          # all_images is comma-separated
        u = u.strip()
        if not u:
            continue
        host = urlparse(u).hostname
        if host:
            hosts[host.lower()] += 1

if "image_url" in df.columns:
    df["image_url"].apply(collect)
if "all_images" in df.columns:
    df["all_images"].apply(collect)

print(f"\nFound {len(hosts)} unique hostname(s) across {len(df)} products:\n")
for host, count in hosts.most_common():
    print(f"  {host:40s}  {count:>6} URLs")

print("\n👉 Put the base domain(s) you see above into:")
print('   CONFIG["image_proxy_allowed_domains"] = [...]')
print("   e.g. if you see 'img.modanisacdn.com', add 'modanisacdn.com'")
print("   (the code matches subdomains automatically via endswith check)")