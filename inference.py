from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))



@dataclass
class InferenceItem:
    id: int
    lat: float
    lon: float


class InferenceDataset(Dataset):
    def __init__(self, annotations_path: str | Path, image_dir: str | Path, transform=None):
        self.annotations_path = Path(annotations_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        with self.annotations_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"{annotations_path} must be a JSON list.")

        self.items: list[InferenceItem] = []
        for x in raw:
            img_path = self.image_dir / f"{x['id']}.jpg"
            if not img_path.exists():
                continue
            self.items.append(
                InferenceItem(
                    id=int(x["id"]),
                    lat=float(x["lat"]),
                    lon=float(x["lon"]),
                )
            )

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items found in {annotations_path} for {image_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(self.image_dir / f"{item.id}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "id": item.id,
            "lat": item.lat,
            "lon": item.lon,
        }


# ---------- Extraction d'embeddings ----------

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[InferenceItem]]:

    model.eval()

    all_embs = []
    all_meta: list[InferenceItem] = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        emb = model(imgs)
        emb = F.normalize(emb, dim=1)
        all_embs.append(emb.cpu())

        bs = len(batch["id"])
        for i in range(bs):
            all_meta.append(
                InferenceItem(
                    id=int(batch["id"][i]),
                    lat=float(batch["lat"][i]),
                    lon=float(batch["lon"][i]),
                )
            )

    return torch.cat(all_embs, dim=0), all_meta


# ---------- Retrieval + métriques ----------

def retrieve_topk(query_embs: torch.Tensor, db_embs: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retourne (scores, indices) où indices[q] contient les k indices database
    les plus proches de query_embs[q] (cosine similarity).
    """
    # query_embs, db_embs sont normalisés → produit scalaire = cosinus
    sims = query_embs @ db_embs.T  # [num_queries, num_db]
    scores, indices = torch.topk(sims, k=k, dim=1)
    return scores, indices


def evaluate_recall(
    query_meta: list[InferenceItem],
    db_meta: list[InferenceItem],
    topk_indices: torch.Tensor,
    k_values: tuple[int, ...],
    distance_thresh_m: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Calcule Recall@K à un seuil distance_thresh_m et construit des résultats détaillés
    par requête, comme recommandé en VPR [web:134][web:160][web:162].
    """
    results: list[dict[str, Any]] = []
    hits = {k: 0 for k in k_values}

    for qi, neigh_idxs in enumerate(topk_indices):
        q = query_meta[qi]
        retrieved = []

        for rank, db_idx in enumerate(neigh_idxs.tolist(), start=1):
            cand = db_meta[db_idx]
            dist = haversine(q.lat, q.lon, cand.lat, cand.lon)
            retrieved.append(
                {
                    "rank": rank,
                    "id": cand.id,
                    "lat": cand.lat,
                    "lon": cand.lon,
                    "distance_m": dist,
                }
            )

        row: dict[str, Any] = {
            "query_id": q.id,
            "gt_lat": q.lat,
            "gt_lon": q.lon,
            "topk": retrieved,
        }

        for k in k_values:
            ok = any(x["distance_m"] <= distance_thresh_m for x in retrieved[:k])
            row[f"success_at_{k}"] = ok
            if ok:
                hits[k] += 1

        results.append(row)

    metrics = {f"recall@{k}": hits[k] / len(query_meta) for k in k_values}
    return results, metrics


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference script for MegaLoc retrieval.")
    p.add_argument("--db-annotations", type=Path, required=True,
                   help="JSON annotations for database images.")
    p.add_argument("--db-images", type=Path, required=True,
                   help="Directory with database images (id.jpg).")
    p.add_argument("--query-annotations", type=Path, required=True,
                   help="JSON annotations for query images.")
    p.add_argument("--query-images", type=Path, required=True,
                   help="Directory with query images (id.jpg).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional fine-tuned model checkpoint (.pth).")
    p.add_argument("--output-dir", type=Path, default=Path("inference_outputs"),
                   help="Where to save results and metrics.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--recall-ks", type=int, nargs="+", default=[1, 5, 10],
                   help="K values for Recall@K.")
    p.add_argument("--distance-thresh-m", type=float, default=25.0,
                   help="Distance threshold in meters for a correct match.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu'.")
    p.add_argument("--multi-gpu", action="store_true",
                   help="Use DataParallel over all available GPUs.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms (do the same as training / MegaLoc expected input)
    tf = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
    ])

    # 1) Datasets + loaders
    db_dataset = InferenceDataset(args.db_annotations, args.db_images, transform=tf)
    query_dataset = InferenceDataset(args.query_annotations, args.query_images, transform=tf)

    db_loader = DataLoader(
        db_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2) Load model
    print("Loading MegaLoc model...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    model = model.to(device)

    if args.checkpoint is not None and args.checkpoint.is_file():
        print(f"Loading checkpoint from {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        # Support both full checkpoints (with "model_state") and raw state_dict.
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    # Optionally wrap in DataParallel for multi-GPU inference
    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # 3) Extract embeddings
    print("Extracting database embeddings...")
    db_embs, db_meta = extract_embeddings(model, db_loader, device)

    print("Extracting query embeddings...")
    query_embs, query_meta = extract_embeddings(model, query_loader, device)

    # 4) Retrieval
    print("Performing retrieval...")
    _, topk_indices = retrieve_topk(query_embs, db_embs, k=args.topk)

    # 5) Metrics
    print("Computing metrics...")
    k_values = tuple(args.recall_ks)
    results, metrics = evaluate_recall(
        query_meta=query_meta,
        db_meta=db_meta,
        topk_indices=topk_indices,
        k_values=k_values,
        distance_thresh_m=args.distance_thresh_m,
    )

    # 6) Save
    results_path = args.output_dir / "retrieval_results.json"
    metrics_path = args.output_dir / "metrics.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieval results to {results_path}")
    print(f"Saved metrics to {metrics_path}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()