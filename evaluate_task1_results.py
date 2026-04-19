import argparse
import json
from pathlib import Path

import config
from utils import angle_diff_deg, haversine_m, load_embeddings_metadata, load_topk_results



def is_correct_match(qm, dm, distance_thresh_m, angle_thresh_deg=None, require_same_sequence=False):
    """
    Determine if a retrieved candidate is a correct match based on distance 
    (and optional angle/sequence criteria).

    Args:
        qm: Query metadata dict with keys 'lat', 'lon', 'angle'
        dm: Database metadata dict with keys 'lat', 'lon', 'angle'
        distance_thresh_m: Distance threshold in meters
        angle_thresh_deg: Angle difference threshold in degrees
        require_same_sequence: If True, also require that query and db items have the same sequence
    """
    dist = haversine_m(qm["lat"], qm["lon"], dm["lat"], dm["lon"])
    if dist > distance_thresh_m:
        return False, dist, None

    angle_gap = angle_diff_deg(qm["angle"], dm["angle"])

    if angle_thresh_deg is not None and angle_gap > angle_thresh_deg:
        return False, dist, angle_gap

    if require_same_sequence and qm.get("sequence_id") != dm.get("sequence_id"):
        return False, dist, angle_gap

    return True, dist, angle_gap


def evaluate(topk_results, db_meta, query_meta, ks, distance_thresh_m, angle_thresh_deg=None, require_same_sequence=False):
    """
    Evaluate retrieval results and compute Recall@K metrics.

    Args:
        topk_results: Dict mapping query_id to list of retrieved items (each with 'id' and 'score')
        db_meta: Dict mapping db_id to metadata dict (with 'lat', 'lon', 'angle', etc.)
        query_meta: Dict mapping query_id to metadata dict
        ks: List of K values for Recall@K (e.g., [1, 5, 10] for Recall@1, Recall@5, Recall@10)
        distance_thresh_m: Distance threshold in meters for a correct match

        angle_thresh_deg: Optional angle difference threshold in degrees for a correct match
        require_same_sequence: If True, also require that query and db items have the same sequence 
    """
    query_ids = sorted(topk_results.keys())
    total = len(query_ids)
    if total == 0:
        raise ValueError("No query found in top-k results.")

    # initialize global hit counters for each K, and a list to store detailed results per query
    hits = {k: 0 for k in ks}
    details = []

    for qid in query_ids:
        if qid not in query_meta:
            continue

        qm = query_meta[qid]
        query_result = {
            "query_id": qid,
            "gt_lat": qm["lat"],
            "gt_lon": qm["lon"],
            "topk": [],
        }

        # Precompute correctness for each ranked retrieval.
        ranked_correct = []

        # iterate through retrieved data candidates for this query and evaluate their correctness
        for rank, item in enumerate(topk_results[qid], start=1):
            rid = int(item["id"])
            score = float(item["score"])

            # retrieve metadata for the retrieved data item, otherwise "skip" it
            dm = db_meta.get(rid)
            if dm is None:
                ranked_correct.append(False)
                query_result["topk"].append(
                    {
                        "rank": rank,
                        "id": rid,
                        "score": score,
                        "error": "missing_db_metadata",
                    }
                )
                continue

            # check if the item is within range of the query (distance, potentially angle and sequence)
            status, dist_m, angle_gap = is_correct_match(
                qm=qm,
                dm=dm,
                distance_thresh_m=distance_thresh_m,
                angle_thresh_deg=angle_thresh_deg,
                require_same_sequence=require_same_sequence,
            )
            ranked_correct.append(status)

            # build the detailed entry for this retrieved item
            entry = {
                "rank": rank,
                "id": rid,
                "score": score,
                "lat": dm["lat"],
                "lon": dm["lon"],
                "distance_m": dist_m,
                "is_correct": status,
            }
            if angle_gap is not None:
                entry["angle_diff_deg"] = angle_gap
            query_result["topk"].append(entry)

        # compute hits for each K based on the ranked correctness
        for k in ks:
            success = any(ranked_correct[:k])
            query_result[f"success_at_{k}"] = success
            if success:
                hits[k] += 1

        details.append(query_result)

    total_queries_num = len(details)
    if total_queries_num == 0:
        raise ValueError("No evaluable query: query metadata missing for all queries.")

    # compute final metrics for all queries
    metrics = {f"recall@{k}": hits[k] / total_queries_num for k in ks}
    metrics["num_queries"] = total_queries_num
    metrics["distance_threshold_m"] = distance_thresh_m
    if angle_thresh_deg is not None:
        metrics["angle_threshold_deg"] = angle_thresh_deg
    if require_same_sequence:
        metrics["require_same_sequence"] = True

    return metrics, details


def main():
    # Main arguments parsing
    parser = argparse.ArgumentParser(description="Evaluate Task1 retrieval results (Recall@K)")
    parser.add_argument("--results", default=config.EVAL_RESULTS, help="Path to outputs/topk_results.json")
    parser.add_argument("--embeddings", default=config.EVAL_EMBEDDINGS, help="Path to outputs/embeddings.pt")
    parser.add_argument("--ks", type=int, nargs="+", default=config.EVAL_KS, help="K values for Recall@K (e.g., --ks 1 5 10 for Recall@1, Recall@5, Recall@10)")
    parser.add_argument("--distance-threshold-m", type=float, default=config.EVAL_DISTANCE_THRESHOLD_M, help="Distance threshold (meters)")
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=config.EVAL_ANGLE_THRESHOLD_DEG,
        help="Optional angle threshold (degrees) for a positive match",
    )
    parser.add_argument(
        "--require-same-sequence",
        action="store_true",
        help="If set, positive match also requires same sequence_id",
    )
    parser.add_argument("--output", default=config.EVAL_OUTPUT, help="Where to save metrics JSON")
    parser.add_argument(
        "--details-output",
        default=config.EVAL_DETAILS_OUTPUT,
        help="Where to save per-query detailed evaluation JSON",
    )
    parser.add_argument(
        "--strict-k",
        action="store_true",
        help="Fail if a requested K is larger than retrieved list length",
    )
    args = parser.parse_args()


    # loading data
    topk_results = load_topk_results(args.results)
    db_meta, query_meta = load_embeddings_metadata(args.embeddings)

    # check that the requested K (size) exists in the given results
    max_retrieved = max((len(v) for v in topk_results.values()), default=0)
    max_k = max(args.ks)
    if max_k > max_retrieved:
        msg = (
            f"Requested max K={max_k}, but results only contain up to {max_retrieved} retrieved candidates. "
            "For a true Recall@10, run task1.py with --topk 10 (or more)."
        )
        if args.strict_k:
            raise ValueError(msg)
        print(f"WARNING: {msg}")

    # calculate metrics and save outputs
    metrics, details = evaluate(
        topk_results=topk_results,
        db_meta=db_meta,
        query_meta=query_meta,
        ks=sorted(set(args.ks)),
        distance_thresh_m=args.distance_threshold_m,
        angle_thresh_deg=args.angle_threshold_deg,
        require_same_sequence=args.require_same_sequence,
    )

    # save metrics and details to JSON files
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    details_output_path = Path(args.details_output)
    details_output_path.parent.mkdir(parents=True, exist_ok=True)
    with details_output_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")
    print(f"Saved per-query details to {details_output_path}")


if __name__ == "__main__":
    main()
