import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from tqdm import tqdm

import config
from geo_triplet_dataset import AnnotationStore
from utils import haversine_m, load_annotation_metadata, save_json



class AnnotationImageDataset(Dataset):
    """Dataset that loads images from annotations and returns image tensors with ids."""

    def __init__(self, annotations_path, image_dir, transform=None):
        """Initialize the dataset from an annotation file and an image directory."""
        self.store = AnnotationStore(annotations_path, image_dir)
        self.transform = transform

        valid_items = []
        invalid_count = 0
        for ann in self.store.items:
            image_path = self.store.image_dir / f"{ann.id}.jpg"
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_items.append(ann)
            except (UnidentifiedImageError, OSError):
                invalid_count += 1

        self.store.items = valid_items
        if invalid_count > 0:
            print(
                f"WARNING: skipped {invalid_count} unreadable image(s) "
                f"from {Path(annotations_path).name}"
            )

    def __len__(self):
        """Return the number of valid samples in the dataset."""
        return len(self.store)

    def __getitem__(self, idx):
        """Return the transformed image and its annotation id for a given index."""
        ann = self.store.items[idx]
        try:
            img = Image.open(self.store.get_image_path(idx)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Skip samples that become unreadable after the initial verification pass.
            return None
        if self.transform is not None:
            img = self.transform(img)
        return img, ann.id


def collate_skip_none(batch):
    """Collate function that drops unreadable samples returned as None."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return default_collate(batch)


def build_transform():
    """Create the image preprocessing pipeline used before inference."""
    return transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
    ])


def extract_descriptor(output):
    """Extract the global descriptor from the model output, regardless of its format."""
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        return output[0]

    if isinstance(output, dict):
        if "global_descriptor" in output:
            return output["global_descriptor"]
        return next(iter(output.values()))

    raise TypeError(f"Unsupported model output type: {type(output)}")


def compute_embeddings(model, device, annotations_path, image_dir, batch_size, num_workers, custom_images=None):
    """Compute and return L2-normalized embeddings for all images in one split.

    Args:
        custom_images: Optional list of image paths to process instead of annotations
    """
    if custom_images:
        # Process specific images without annotations
        from PIL import Image
        transform = build_transform()
        all_ids = []
        all_desc = []

        model.eval()
        with torch.no_grad():
            for img_path in tqdm(custom_images, desc="Embedding custom images"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    desc = extract_descriptor(model(img_tensor))
                    desc = F.normalize(desc, p=2, dim=1)
                    all_desc.append(desc.cpu())
                    # Use filename stem as id
                    img_id = int(Path(img_path).stem) if Path(img_path).stem.isdigit() else hash(Path(img_path).stem) % (2**31)
                    all_ids.append(img_id)
                except Exception as e:
                    print(f"Warning: Could not process {img_path}: {e}")

        embeddings = torch.cat(all_desc, dim=0) if all_desc else torch.empty(0)
        return all_ids, embeddings

    # Original code for annotation-based processing
    dataset = AnnotationImageDataset(
        annotations_path=annotations_path,
        image_dir=image_dir,
        transform=build_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_skip_none,
    )

    all_ids = []
    all_desc = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding {Path(annotations_path).name}", unit="batch"):
            if batch is None:
                continue
            images, ids = batch
            images = images.to(device, non_blocking=True)
            desc = extract_descriptor(model(images))
            desc = F.normalize(desc, p=2, dim=1) 

            all_desc.append(desc.cpu()) # back to CPU for concatenation later
            # DataLoader can collate ids as tensors; we convert them to plain Python ints.
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().tolist())
            else:
                all_ids.extend([int(x) for x in ids])

    # concat the batchs into a single tensor
    embeddings = torch.cat(all_desc, dim=0) if all_desc else torch.empty(0)
    return all_ids, embeddings


def estimate_rank_medoid_position(topk_items, db_meta, estimate_topk):
    """Estimate query position with a rank-weighted medoid over top-k retrieved items."""
    k = min(estimate_topk, len(topk_items))
    candidates = []

    for rank, item in enumerate(topk_items[:k], start=1):
        db_id = int(item["id"])
        meta = db_meta.get(db_id)
        if meta is None:
            continue
        if "lat" not in meta or "lon" not in meta:
            continue

        candidates.append(
            {
                "rank": rank,
                "lat": float(meta["lat"]),
                "lon": float(meta["lon"]),
            }
        )

    if not candidates:
        return None
    if len(candidates) == 1:
        return {"lat": candidates[0]["lat"], "lon": candidates[0]["lon"]}

    # Emphasize higher-ranked neighbors while keeping medoid robustness to outliers.
    rank_weights = {c["rank"]: 1.0 / math.log2(c["rank"] + 1.0) for c in candidates}
    best_idx = 0
    best_cost = float("inf")

    for i, cand_i in enumerate(candidates):
        total_cost = 0.0
        for cand_j in candidates:
            d = haversine_m(cand_i["lat"], cand_i["lon"], cand_j["lat"], cand_j["lon"])
            total_cost += rank_weights[cand_j["rank"]] * d

        if total_cost < best_cost:
            best_cost = total_cost
            best_idx = i

    best = candidates[best_idx]
    return {"lat": best["lat"], "lon": best["lon"]}


def retrieve_topk(db_ids, db_emb, q_ids, q_emb, db_meta, topk=5, estimate_topk=5):
    """Retrieve top-k items and a Top-5 rank-medoid position estimate per query."""
    # cosine similarity to rank database items for each query
    sim = q_emb @ db_emb.T
    # Ensure sim is 2D: shape should be [num_queries, num_database_items]
    if sim.dim() == 0:
        # Empty case - no queries or no database items
        return {}
    elif sim.dim() == 1:
        # Single query case - shape [num_database_items] -> [1, num_database_items]
        sim = sim.unsqueeze(0)
    # Now sim is guaranteed to be 2D
    k = min(topk, sim.shape[1])
    results = {}
    method_name = f"rank_medoid_top{int(estimate_topk)}"

    for i, qid in enumerate(q_ids):
        # get the top-k most similar database items for this query
        scores, db_idx = torch.topk(sim[i], k=k)
        topk_items = [
            {"id": int(db_ids[idx]), "score": float(scores[rank].item())}
            for rank, idx in enumerate(db_idx.tolist())
        ]
        estimate = estimate_rank_medoid_position(
            topk_items=topk_items,
            db_meta=db_meta,
            estimate_topk=estimate_topk,
        )

        results[int(qid)] = {
            "topk": topk_items,
            "position_estimates": {
                method_name: estimate,
            },
        }
    return results


def main():
    """Parse arguments, run inference, retrieve top-k results, and save outputs."""
    parser = argparse.ArgumentParser(description="Task1 inference + top-k retrieval with MegaLoc")
    parser.add_argument("--image-dir", default=config.IMAGE_DIR)
    parser.add_argument("--annotations-db", default=config.TASK1_ANNOTATIONS_DB)
    parser.add_argument("--annotations-query", default=config.TASK1_ANNOTATIONS_QUERY)
    parser.add_argument("--checkpoint", default=config.TASK1_CHECKPOINT)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=config.TASK1_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.TASK1_NUM_WORKERS)
    parser.add_argument("--topk", type=int, default=config.TASK1_TOPK)
    parser.add_argument(
        "--position-estimation-topk",
        type=int,
        default=5,
        help="Number of top retrieved neighbors used for rank-medoid position estimation",
    )
    parser.add_argument("--output", default=config.TASK1_OUTPUT)
    parser.add_argument("--embeddings-output", default=config.TASK1_EMBEDDINGS_OUTPUT)
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    db_meta = load_annotation_metadata(args.annotations_db, args.image_dir)
    query_meta = load_annotation_metadata(args.annotations_query, args.image_dir)

    # Compute embeddings for database and query splits
    db_ids, db_emb = compute_embeddings(
        model=model,
        device=device,
        annotations_path=args.annotations_db,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Compute query embeddings and retrieve top-k results
    q_ids, q_emb = compute_embeddings(
        model=model,
        device=device,
        annotations_path=args.annotations_query,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = retrieve_topk(
        db_ids=db_ids,
        db_emb=db_emb,
        q_ids=q_ids,
        q_emb=q_emb,
        db_meta=db_meta,
        topk=args.topk,
        estimate_topk=args.position_estimation_topk,
    )

    save_json(results, args.output)

    torch.save(
        {
            "database": {"ids": db_ids, "embeddings": db_emb},
            "queries": {"ids": q_ids, "embeddings": q_emb},
            "db_meta": db_meta,
            "query_meta": query_meta,
        },
        args.embeddings_output,
    )

    print(f"Saved retrieval results to {args.output}")
    print(f"Saved embeddings to {args.embeddings_output}")


if __name__ == "__main__":
    main()