"""Fix inverted train/val splits for all cities in the dataset.

The dataset_builder used to produce 80% validation and 20% train because
VAL_RATIO was set to 0.8. This script swaps the split files so that the
training set is the larger one and the validation set is the smaller one.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def fix_city(city_dir, dry_run=False):
    """Swap train and val metadata for a single city and update split_stats."""
    train_path = Path(city_dir) / "metadata_train.json"
    val_path = Path(city_dir) / "metadata_val.json"
    stats_path = Path(city_dir) / "split_stats.json"

    if not train_path.exists() or not val_path.exists():
        print(f"Skipping {city_dir}: train/val files not found")
        return

    train_data = load_json(train_path)
    val_data = load_json(val_path)

    old_stats = load_json(stats_path) if stats_path.exists() else {}

    print(f"{city_dir}: swapping train ({len(train_data)}) <-> val ({len(val_data)})")

    if not dry_run:
        # Back up original files
        shutil.copy2(train_path, train_path.with_suffix(".json.bak"))
        shutil.copy2(val_path, val_path.with_suffix(".json.bak"))

        # Swap metadata
        save_json(train_path, val_data)
        save_json(val_path, train_data)

        # Update split stats
        new_stats = {
            "num_train_images": old_stats.get("num_val_images", len(val_data)),
            "num_train_sequences": old_stats.get("num_val_sequences", 0),
            "num_val_images": old_stats.get("num_train_images", len(train_data)),
            "num_val_sequences": old_stats.get("num_train_sequences", 0),
        }
        save_json(stats_path, new_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Swap inverted train/val metadata files for all cities."
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Root dataset directory containing city subdirectories (default: dataset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be swapped without modifying files.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    for city_dir in sorted(dataset_dir.iterdir()):
        if city_dir.is_dir():
            fix_city(city_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No files were modified.")
    else:
        print("\nTrain/val splits fixed for all cities.")


if __name__ == "__main__":
    main()
