import json
import pathlib
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

CATEGORIES = [
    {"name": "Category:Microscopy", "directory": "Microscopy"},
    {"name": "Category:Scientific instruments", "directory": "Scientific Instruments"},
    {"name": "Category:Astronomy", "directory": "Astronomy"},
    {"name": "Category:Geology", "directory": "Geology"},
    {"name": "Category:Historical maps", "directory": "Historical Maps"},
    {"name": "Category:Archaeology", "directory": "Archaeology"},
    {"name": "Category:Manuscripts", "directory": "Manuscripts"},
    {"name": "Category:Historical photographs", "directory": "Historical Photographs"},
    {"name": "Category:Military history", "directory": "Military History"},
    {"name": "Category:Historical technology", "directory": "Historical Technology"},
    {"name": "Category:Topographic maps", "directory": "Topographic Maps"},
    {"name": "Category:Satellite images", "directory": "Satellite Images"},
    {"name": "Category:Physical geography", "directory": "Physical Geography"},
    {"name": "Category:Volcanoes", "directory": "Volcanoes"},
    {"name": "Category:Glaciers", "directory": "Glaciers"},
    {"name": "Category:Botanical illustrations", "directory": "Botanical Illustrations"},
    {"name": "Category:Herbarium specimens", "directory": "Herbarium Specimens"},
    {"name": "Category:Animal anatomy", "directory": "Animal Anatomy"},
    {"name": "Category:Plant anatomy", "directory": "Plant Anatomy"},
    {"name": "Category:Fossils", "directory": "Fossils"},
    {"name": "Category:Computer architecture", "directory": "Computer Architecture"},
    {"name": "Category:Computer hardware", "directory": "Computer Hardware"},
    {"name": "Category:Computer networks", "directory": "Computer Networks"},
    {"name": "Category:Algorithms", "directory": "Algorithms"},
    {"name": "Category:Neural networks", "directory": "Neural Networks"},
    {"name": "Category:Bridge engineering", "directory": "Bridge Engineering"},
    {"name": "Category:Mechanical engineering", "directory": "Mechanical Engineering"},
    {"name": "Category:Technical drawings", "directory": "Technical Drawings"},
    {"name": "Category:Aircraft", "directory": "Aircraft"},
    {"name": "Category:Robotics", "directory": "Robotics"},
]

IMAGES_PER_CATEGORY = 15
IMAGE_ROOT = pathlib.Path("eval/data/images")
MANIFEST_PATH = IMAGE_ROOT / "manifest.json"
USER_AGENT = "FindMyFiles-evaluation-corpus/0.1"
MAX_RETRIES = 5

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)


session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def load_manifest():
    entries = []
    if MANIFEST_PATH.exists():
        entries.extend(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))

    # Import manifests created by the earlier single-category version.
    for legacy_path in IMAGE_ROOT.glob("*/manifest.json"):
        entries.extend(json.loads(legacy_path.read_text(encoding="utf-8")))

    return {entry["source_url"]: entry for entry in entries}


manifest_by_source = load_manifest()


def metadata_value(metadata, name):
    value = metadata.get(name, {}).get("value")
    return value.strip() if isinstance(value, str) else value


def save_manifest():
    MANIFEST_PATH.write_text(
        json.dumps(
            sorted(manifest_by_source.values(), key=lambda entry: entry["source_url"]),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def get_with_retries(url, *, params=None, timeout=30):
    for attempt in range(MAX_RETRIES):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        if attempt == MAX_RETRIES - 1:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        try:
            wait_seconds = max(1, min(int(retry_after), 60))
        except (TypeError, ValueError):
            wait_seconds = min(2**attempt, 60)

        print(f"Rate limited; retrying in {wait_seconds}s...")
        time.sleep(wait_seconds)

    raise RuntimeError("Request retry loop ended unexpectedly")


def source_url_for(title):
    return "https://commons.wikimedia.org/wiki/" + quote(
        title.replace(" ", "_"), safe=":/"
    )


def category_count(category_name):
    return sum(
        entry.get("category") == category_name
        and pathlib.Path(entry["local_path"]).exists()
        for entry in manifest_by_source.values()
    )


def process_category(category):
    category_name = category["name"]
    output_dir = IMAGE_ROOT / category["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = category_count(category_name)
    continuation = None

    while downloaded_count < IMAGES_PER_CATEGORY:
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": category_name,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "1600",
        }
        if continuation:
            params.update(continuation)

        response = get_with_retries(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            timeout=30,
        )
        data = response.json()

        for page in data.get("query", {}).get("pages", {}).values():
            if downloaded_count >= IMAGES_PER_CATEGORY:
                break

            image_info = page.get("imageinfo", [{}])[0]
            image_url = image_info.get("thumburl") or image_info.get("url")
            page_title = page.get("title", "")
            source_url = source_url_for(page_title)

            if not image_url or source_url in manifest_by_source:
                continue

            filename = pathlib.Path(image_url.split("?")[0]).name
            output_path = output_dir / filename

            if not output_path.exists():
                try:
                    image = get_with_retries(image_url, timeout=60)
                except requests.HTTPError as error:
                    print(f"Skipping {filename}: {error}")
                    continue
                output_path.write_bytes(image.content)
                print(f"Downloaded {output_path}")
            else:
                print(f"Already exists: {output_path}")

            metadata = image_info.get("extmetadata", {})
            manifest_by_source[source_url] = {
                "filename": filename,
                "local_path": str(output_path),
                "title": page_title,
                "source_url": source_url,
                "download_url": image_url,
                "category": category_name,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "author": metadata_value(metadata, "Artist"),
                "license": metadata_value(metadata, "LicenseShortName"),
                "license_url": metadata_value(metadata, "LicenseUrl"),
            }
            downloaded_count += 1
            save_manifest()

        continuation = data.get("continue")
        if not continuation:
            break

    print(f"{category_name}: {downloaded_count}/{IMAGES_PER_CATEGORY} images")


for category in CATEGORIES:
    process_category(category)

save_manifest()
print(f"Manifest written to {MANIFEST_PATH} ({len(manifest_by_source)} images)")
