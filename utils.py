import json
import math
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates in meters."""
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def angle_diff_deg(a, b):
    """Smallest absolute angle difference in degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def load_topk_results(results_path):
    """Load retrieval results JSON and normalize query ids as ints.

    Supports two formats:
    - legacy: {query_id: [{id, score}, ...]}
    - enriched: {query_id: {"topk": [{id, score}, ...], ...}}
    """
    with Path(results_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    normalized = {}
    for k, v in raw.items():
        qid = int(k)
        if isinstance(v, dict) and "topk" in v:
            normalized[qid] = v["topk"]
        else:
            normalized[qid] = v
    return normalized


def load_retrieval_results(results_path):
    """Load retrieval JSON and return a normalized enriched structure.

    Output format:
    {
      query_id: {
        "topk": [{"id": int, "score": float}, ...],
        "position_estimates": {method: {"lat": float, "lon": float} | None}
      }
    }
    """
    with Path(results_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = {}
    for k, v in raw.items():
        qid = int(k)

        if isinstance(v, dict):
            topk = v.get("topk", [])
            pos = v.get("position_estimates", {})
        else:
            # Legacy format: value is directly the top-k list.
            topk = v
            pos = {}

        normalized[qid] = {
            "topk": topk,
            "position_estimates": pos,
        }

    return normalized


def load_embeddings_metadata(embeddings_path):
    """Load db/query metadata saved in embeddings.pt and normalize keys to ints."""
    import torch

    payload = torch.load(embeddings_path, map_location="cpu")
    db_meta = payload.get("db_meta", {})
    query_meta = payload.get("query_meta", {})
    return {int(k): v for k, v in db_meta.items()}, {int(k): v for k, v in query_meta.items()}


def load_annotation_metadata(annotations_path, image_dir):
    """Load annotation metadata and index it by image id."""
    from dataset.geo_triplet_dataset import AnnotationStore

    store = AnnotationStore(annotations_path, image_dir)
    return {
        ann.id: {
            "lat": ann.lat,
            "lon": ann.lon,
            "angle": ann.angle,
            "sequence_id": ann.sequence_id,
        }
        for ann in store.items
    }


def save_json(path, data):
    """Save a Python object as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
