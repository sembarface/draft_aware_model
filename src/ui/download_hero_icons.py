import argparse
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
HEROES_PATH = ROOT_DIR / "data" / "heroes.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "hero_icons"

HERO_IMAGE_KEY_OVERRIDES = {
    "Anti-Mage": "antimage",
    "Centaur Warrunner": "centaur",
    "Clockwerk": "rattletrap",
    "Doom": "doom_bringer",
    "Io": "wisp",
    "Lifestealer": "life_stealer",
    "Magnus": "magnataur",
    "Nature's Prophet": "furion",
    "Necrophos": "necrolyte",
    "Outworld Destroyer": "obsidian_destroyer",
    "Queen of Pain": "queenofpain",
    "Shadow Fiend": "nevermore",
    "Timbersaw": "shredder",
    "Treant Protector": "treant",
    "Underlord": "abyssal_underlord",
    "Vengeful Spirit": "vengefulspirit",
    "Windranger": "windrunner",
    "Wraith King": "skeleton_king",
    "Zeus": "zuus",
}


def hero_image_key(hero_name):
    if not hero_name:
        return ""
    if hero_name in HERO_IMAGE_KEY_OVERRIDES:
        return HERO_IMAGE_KEY_OVERRIDES[hero_name]
    return hero_name.lower().replace("'", "").replace("-", "").replace(" ", "_")


def hero_image_url(hero_key):
    return f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{hero_key}.png"


def load_heroes(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(int(row["id"]), row["name"]) for row in reader]


def download_icon(hero_name, output_dir, force=False, timeout=5):
    key = hero_image_key(hero_name)
    if not key:
        return "failed", hero_name, "missing image key"
    output_path = output_dir / f"{key}.png"
    if output_path.exists() and not force:
        return "skipped", hero_name, str(output_path)
    url = hero_image_url(key)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read()
        if not content:
            return "failed", hero_name, "empty response"
        output_path.write_bytes(content)
        return "downloaded", hero_name, str(output_path)
    except Exception as exc:
        return "failed", hero_name, f"{url} ({exc})"


def download_all(output_dir=DEFAULT_OUTPUT_DIR, force=False, workers=12, timeout=5):
    if not HEROES_PATH.exists():
        raise FileNotFoundError(f"missing heroes table: {HEROES_PATH}")
    output_dir.mkdir(parents=True, exist_ok=True)
    heroes = load_heroes(HEROES_PATH)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(download_icon, name, output_dir, force, timeout): name
            for _, name in heroes
        }
        for future in as_completed(future_to_name):
            results.append(future.result())
    counts = {}
    for status, _, _ in results:
        counts[status] = counts.get(status, 0) + 1
    print(f"hero icons directory: {output_dir}")
    print(
        "summary: "
        f"downloaded={counts.get('downloaded', 0)}, "
        f"skipped={counts.get('skipped', 0)}, "
        f"failed={counts.get('failed', 0)}"
    )
    failed = [item for item in results if item[0] == "failed"]
    if failed:
        print("failed icons:")
        for _, hero_name, message in failed:
            print(f"- {hero_name}: {message}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Download Dota hero icons for the local Streamlit UI.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download icons that already exist.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    return download_all(output_dir=args.output_dir, force=args.force, workers=args.workers, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
