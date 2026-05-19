"""
dataset_collection.py
=====================
Multi-query image scraper for guitar classification dataset.

Uses icrawler (Bing + Google) with many targeted search queries
to collect a large, diverse dataset. Queries are designed to pull
product images from retailer sites (Guitar Center, Sweetwater, etc.)
as well as general guitar photos.

Classes (3):
  - acoustic   (includes classical / nylon guitars)
  - electric
  - bass

Directory structure produced:
    dataset/
    ├── acoustic/
    ├── electric/
    └── bass/

Usage:
    python dataset_collection.py
"""

import os
import hashlib
import time
import logging
from pathlib import Path

from PIL import Image
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler


# Configuration
DATASET_DIR = "dataset"

# Minimum acceptable image dimensions (width, height).
MIN_IMAGE_SIZE = (80, 80)

# Resize all kept images to this size. Set to None to skip.
RESIZE_TO = (256, 256)

# Images per query per search engine.
IMAGES_PER_QUERY = 150

# Many diverse queries per class to maximize image count & diversity.
# Each query targets a different angle/style/brand/site.
CLASSES = {
    "acoustic": [
        # Product-style queries (these pull retailer images)
        "acoustic guitar product photo",
        "acoustic guitar white background",
        "acoustic guitar full body image",
        "acoustic guitar front view",
        "dreadnought guitar product image",
        "acoustic guitar isolated",
        # Classical / nylon (merged into acoustic)
        "classical guitar product photo",
        "classical guitar full body",
        "nylon string guitar product image",
        "classical guitar white background",
        # Brand-specific (acoustic brands)
        "Taylor acoustic guitar",
        "Martin acoustic guitar",
        "Yamaha acoustic guitar",
        "Fender acoustic guitar",
        "Gibson acoustic guitar",
        "Epiphone acoustic guitar",
        "Takamine acoustic guitar",
        # Style variations
        "acoustic electric guitar product",
        "12 string acoustic guitar",
        "parlor guitar product photo",
        "jumbo acoustic guitar",
        "concert acoustic guitar",
        "cutaway acoustic guitar",
    ],
    "electric": [
        # Product-style queries
        "electric guitar product photo",
        "electric guitar white background",
        "electric guitar full body image",
        "electric guitar front view",
        "electric guitar isolated",
        "solid body electric guitar",
        # Shape-specific
        "stratocaster electric guitar",
        "telecaster electric guitar",
        "les paul electric guitar",
        "SG electric guitar",
        "flying V electric guitar",
        "explorer electric guitar",
        "superstrat electric guitar",
        "semi hollow electric guitar",
        "hollow body electric guitar",
        # Brand-specific
        "Fender electric guitar",
        "Gibson electric guitar",
        "Ibanez electric guitar",
        "PRS electric guitar",
        "ESP electric guitar",
        "Jackson electric guitar",
        "Schecter electric guitar",
        "Epiphone electric guitar",
        "Squier electric guitar",
    ],
    "bass": [
        # Product-style queries
        "bass guitar product photo",
        "bass guitar white background",
        "bass guitar full body image",
        "bass guitar front view",
        "bass guitar isolated",
        "electric bass guitar",
        # String count
        "4 string bass guitar",
        "5 string bass guitar",
        "6 string bass guitar",
        # Shape/style
        "precision bass guitar",
        "jazz bass guitar",
        "P bass guitar product",
        "J bass guitar product",
        "modern bass guitar",
        # Brand-specific
        "Fender bass guitar",
        "Ibanez bass guitar",
        "Music Man bass guitar",
        "Yamaha bass guitar",
        "Squier bass guitar",
        "Spector bass guitar",
        "Schecter bass guitar",
        "Rickenbacker bass guitar",
        "ESP bass guitar",
    ],
}

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



# Utility Functions



def create_directories() -> None:
    """Create the dataset directory structure."""
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        log.info(f"Directory ready: {class_dir}")
    print()


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(filepath: str) -> bool:
    """Check if a file is a valid, non-corrupt image meeting size requirements."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            w, h = img.size
            if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
                return False
        return True
    except Exception:
        return False


def resize_image(filepath: str, size: tuple) -> None:
    """Resize an image and save as JPEG."""
    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            img.save(filepath, "JPEG", quality=90)
    except Exception as e:
        log.warning(f"Resize failed for {filepath}: {e}")


def count_images(class_name: str) -> int:
    d = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if not f.startswith(".")])



# Scraping



def scrape_class(class_name: str, queries: list) -> None:
    class_dir = os.path.join(DATASET_DIR, class_name)

    
    print(f"SCRAPING: {class_name.upper()}  ({len(queries)} queries)")
    

    for idx, query in enumerate(queries, start=1):
        print(f"[{idx}/{len(queries)}] \"{query}\"")

        # Bing
        try:
            bing = BingImageCrawler(
                storage={"root_dir": class_dir},
                downloader_threads=4,
                log_level="WARNING",
            )
            bing.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                file_idx_offset="auto",
            )
            print(f"Bing ✓ (total so far: {count_images(class_name)})")
        except Exception as e:
            print(f"Bing ✗ {e}")

        # Google
        try:
            google = GoogleImageCrawler(
                storage={"root_dir": class_dir},
                downloader_threads=4,
                log_level="WARNING",
            )
            google.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                file_idx_offset="auto",
            )
            print(f"Google ✓ (total so far: {count_images(class_name)})")
        except Exception as e:
            print(f"Google ✗ {e}")

        time.sleep(1.5)

    total = count_images(class_name)
    print(f">> Raw images for '{class_name}': {total}")



# Post-processing



def clean_class_directory(class_name: str) -> dict:
    class_dir = os.path.join(DATASET_DIR, class_name)
    files = sorted(Path(class_dir).iterdir())

    stats = {
        "total_before": len(files),
        "removed_corrupt": 0,
        "removed_duplicate": 0,
        "kept": 0,
    }
    seen_hashes = set()

    for filepath in files:
        fp = str(filepath)

        if not is_valid_image(fp):
            os.remove(fp)
            stats["removed_corrupt"] += 1
            continue

        fhash = compute_file_hash(fp)
        if fhash in seen_hashes:
            os.remove(fp)
            stats["removed_duplicate"] += 1
            continue
        seen_hashes.add(fhash)

        if RESIZE_TO is not None:
            resize_image(fp, RESIZE_TO)

        stats["kept"] += 1

    return stats


def clean_dataset() -> None:
    
    print("POST-PROCESSING: Cleaning dataset")
    

    total = 0
    for class_name in CLASSES:
        stats = clean_class_directory(class_name)
        total += stats["kept"]
        print(
            f"{class_name.upper():<12} | "
            f"Before: {stats['total_before']:>5} | "
            f"Corrupt: {stats['removed_corrupt']:>4} | "
            f"Duplicate: {stats['removed_duplicate']:>4} | "
            f"Kept: {stats['kept']:>5}"
        )

    print(f">> Total usable images: {total}")
    print(f">> Dataset location:    ./{DATASET_DIR}/")



# Main



def main():
    print("GUITAR IMAGE DATASET COLLECTION")
    print("Sources: Bing + Google Image Search")
    print("Classes: acoustic (+ classical), electric, bass")

    # Step 1: Create directories
    create_directories()

    # Step 2: Scrape all classes
    for class_name, queries in CLASSES.items():
        scrape_class(class_name, queries)

    # Step 3: Clean the dataset
    clean_dataset()

    # Step 4: Final summary
    print("=" * 60)
    print("FINAL COUNTS")
    print("=" * 60)
    for class_name in CLASSES:
        n = count_images(class_name)
        print(f"{class_name:<12}: {n:>5} images")
    print("DONE ")


if __name__ == "__main__":
    main()
