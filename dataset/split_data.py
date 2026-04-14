"""
 Split data into train and validation sets
"""

from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_annotations(json_path: Path) -> list[dict[str, Any]]:
    """ Loads annotations from a JSON file and ensures it's a list of dicts """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a JSON list of annotations.")

    return data


def validate_annotations(data: list[dict[str, Any]], required_keys: set[str]) -> None:
    """ Validates that each annotation is a dict and contains required keys """
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Annotation at index {i} is not a JSON object.")

        missing = required_keys - item.keys()
        if missing:
            raise ValueError(
                f"Annotation at index {i} is missing keys: {sorted(missing)}"
            )


def group_by_sequence(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """ Groups annotations by their sequence_id for sequence-level splitting (and not image level) """
    grouped = defaultdict(list)
    for item in data:
        grouped[str(item["sequence_id"])].append(item)
    return dict(grouped)


def split_sequence_ids(
    sequence_ids: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """ Splits sequence IDs into train and validation sets based on the specified ratio and random seed """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("validation ratio must be between 0 and 1.")

    rng = random.Random(seed)
    seqs = sequence_ids.copy()
    rng.shuffle(seqs)

    n_val = max(1, int(len(seqs) * val_ratio))
    val_seq_ids = set(seqs[:n_val]) # val seq ids are the first n_val after shuffling
    train_seq_ids = set(seqs[n_val:])

    if len(train_seq_ids) == 0:
        raise ValueError("Train split is empty. Reduce val_ratio.")

    return train_seq_ids, val_seq_ids


def build_splits(
    grouped: dict[str, list[dict[str, Any]]],
    train_seq_ids: set[str],
    val_seq_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """ Builds train and validation splits by collecting annotations from sequences assigned to each split """
    train_data = []
    val_data = []

    for seq_id, items in grouped.items():
        if seq_id in train_seq_ids:
            train_data.extend(items)
        elif seq_id in val_seq_ids:
            val_data.extend(items)
        else:
            raise RuntimeError(f"Sequence {seq_id} was not assigned to any split.")

    return train_data, val_data


def compute_stats(
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
) -> dict[str, int]:
    """ Computes statistics about the splits, such as number of images and sequences in each split """
    train_sequences = {str(x["sequence_id"]) for x in train_data}
    val_sequences = {str(x["sequence_id"]) for x in val_data}

    return {
        "num_train_images": len(train_data),
        "num_val_images": len(val_data),
        "num_train_sequences": len(train_sequences),
        "num_val_sequences": len(val_sequences),
    }


def save_json(data: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_stats(stats: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = ArgumentParser(
        description="Split annotations JSON into train/val by sequence_id."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/scratch/users/agraillet/images/metadata.json"),
        help="Path to the input annotations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Directory where split files will be written",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio at sequence level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    required_keys = {"id", "lat", "lon", "angle", "sequence_id"}

    data = load_annotations(args.input)
    validate_annotations(data, required_keys)

    grouped = group_by_sequence(data)
    sequence_ids = list(grouped.keys())

    train_seq_ids, val_seq_ids = split_sequence_ids(
        sequence_ids=sequence_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_data, val_data = build_splits(
        grouped=grouped,
        train_seq_ids=train_seq_ids,
        val_seq_ids=val_seq_ids,
    )

    stats = compute_stats(train_data, val_data)

    save_json(train_data, args.output_dir / "annotations_train.json")
    save_json(val_data, args.output_dir / "annotations_val.json")
    save_stats(stats, args.output_dir / "split_stats.json")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()