import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config
from dataset.geo_triplet_dataset import AnnotationStore
from utils import load_annotation_metadata, save_json



class AnnotationImageDataset(Dataset):
    """Dataset that loads images from annotations and returns image tensors with ids."""

    def __init__(self, annotations_path, image_dir, transform=None):
        """Initialize the dataset from an annotation file and an image directory."""
        self.store = AnnotationStore(annotations_path, image_dir)
        self.transform = transform

    def __len__(self):
        """Return the number of valid samples in the dataset."""
        return len(self.store)

    def __getitem__(self, idx):
        """Return the transformed image and its annotation id for a given index."""
        ann = self.store.items[idx]
        img = Image.open(self.store.get_image_path(idx)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ann.id


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


def compute_embeddings(model, device, annotations_path, image_dir, batch_size, num_workers):
    """Compute and return L2-normalized embeddings for all images in one split."""
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
    )

    all_ids = []
    all_desc = []

    model.eval()
    with torch.no_grad():
        for images, ids in tqdm(loader, desc=f"Embedding {Path(annotations_path).name}", unit="batch"):
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


def retrieve_topk(db_ids, db_emb, q_ids, q_emb, topk=5):
    """Retrieve the top-k most similar database images for each query embedding."""
    # cosine similarity to rank database items for each query
    sim = q_emb @ db_emb.T
    k = min(topk, sim.shape[1])
    results = {}

    for i, qid in enumerate(q_ids):
        # get the top-k most similar database items for this query
        scores, db_idx = torch.topk(sim[i], k=k)

        results[int(qid)] = [
            {"id": int(db_ids[idx]), "score": float(scores[rank].item())}
            for rank, idx in enumerate(db_idx.tolist())
        ]
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

    results = retrieve_topk(db_ids, db_emb, q_ids, q_emb, topk=args.topk)

    save_json(args.output, results)

    torch.save(
        {
            "database": {"ids": db_ids, "embeddings": db_emb},
            "queries": {"ids": q_ids, "embeddings": q_emb},
            "db_meta": load_annotation_metadata(args.annotations_db, args.image_dir),
            "query_meta": load_annotation_metadata(args.annotations_query, args.image_dir),
        },
        args.embeddings_output,
    )

    print(f"Saved retrieval results to {args.output}")
    print(f"Saved embeddings to {args.embeddings_output}")


if __name__ == "__main__":
    main()