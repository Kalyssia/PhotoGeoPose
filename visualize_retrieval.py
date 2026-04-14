from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize MegaLoc retrieval results.")
    p.add_argument("--results", type=Path, required=True, help="Path to retrieval_results.json.")
    p.add_argument("--image-dir", type=Path, required=True, help="Directory containing images (id.jpg).")
    p.add_argument("--output-dir", type=Path, default=Path("retrieval_viz"), help="Directory to save visualizations.")
    p.add_argument("--topk", type=int, default=5, help="Number of top retrieved images to show.")
    p.add_argument("--max-examples", type=int, default=50, help="Maximum number of queries to visualize.")
    p.add_argument("--distance-thresh-m", type=float, default=25.0, help="Distance threshold in meters to mark hits.")
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image(image_dir: Path, image_id: int) -> Image.Image:
    path = image_dir / f"{image_id}.jpg"
    return Image.open(path).convert("RGB")


def visualize_example(
    example: dict,
    image_dir: Path,
    output_path: Path,
    topk: int,
    distance_thresh_m: float,
) -> None:
    query_id = example["query_id"]
    retrieved = example["topk"][:topk]

    n_cols = 1 + len(retrieved)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    query_img = load_image(image_dir, query_id)
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query {query_id}")
    axes[0].axis("off")

    for i, cand in enumerate(retrieved, start=1):
        cid = cand["id"]
        dist = float(cand["distance_m"])
        hit = dist <= distance_thresh_m

        try:
            img = load_image(image_dir, cid)
        except FileNotFoundError:
            axes[i].set_visible(False)
            continue

        axes[i].imshow(img)
        title = f"rank {cand['rank']}\nid {cid}\n{dist:.1f} m"
        if hit:
            title += " (hit)"
        axes[i].set_title(title)
        axes[i].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    results = load_json(args.results)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_examples = min(len(results), args.max_examples)

    for idx in range(n_examples):
        example = results[idx]
        query_id = example["query_id"]
        out_path = output_dir / f"query_{query_id}_top{args.topk}.png"
        visualize_example(
            example=example,
            image_dir=args.image_dir,
            output_path=out_path,
            topk=args.topk,
            distance_thresh_m=args.distance_thresh_m,
        )


if __name__ == "__main__":
    main()
