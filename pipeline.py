"""Combined pipeline: Task 1 (MegaLoc retrieval) + Task 2 (LightGlue angle estimation).

This script runs a full pipeline that:
1. Runs Task 1 to retrieve top-k matches for each query
2. Runs Task 2 to estimate angles using retrieved candidates
3. Outputs YAML with combined evaluation metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from lightglue import LightGlue, SuperPoint
from tqdm import tqdm

import config

from task1 import (
    compute_embeddings,
    retrieve_topk,
)

from task2 import estimate_query_orientation

from evaluate_task1_results import evaluate as evaluate_task1

from utils import (
    angle_diff_deg,
    haversine_m,
    load_annotation_metadata,
    load_embeddings_metadata,
    save_json,
)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj


# Pipeline Main Functions

def run_task1(args, device):
    """Run Task 1 retrieval."""
    print("\nTASK 1: MegaLoc Retrieval\n")

    # Load MegaLoc model
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Load metadata
    db_meta = load_annotation_metadata(args.annotations_db, args.image_dir)
    query_meta = load_annotation_metadata(args.annotations_query, args.image_dir)

    # Compute embeddings
    db_ids, db_emb = compute_embeddings(
        model=model,
        device=device,
        annotations_path=args.annotations_db,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    q_ids, q_emb = compute_embeddings(
        model=model,
        device=device,
        annotations_path=args.annotations_query,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Retrieve top-k
    results = retrieve_topk(
        db_ids=db_ids,
        db_emb=db_emb,
        q_ids=q_ids,
        q_emb=q_emb,
        db_meta=db_meta,
        topk=args.topk,
        estimate_topk=args.position_estimation_topk,
    )

    # Save Task 1 results
    task1_output = Path(args.output_dir) / f"task1_results_{args.city}.json"
    save_json(str(task1_output), results)
    print(f"Saved Task 1 results to {task1_output}")

    embeddings_output = Path(args.output_dir) / "embeddings.pt"
    torch.save(
        {
            "database": {"ids": db_ids, "embeddings": db_emb},
            "queries": {"ids": q_ids, "embeddings": q_emb},
            "db_meta": db_meta,
            "query_meta": query_meta,
        },
        embeddings_output,
    )
    print(f"Saved embeddings to {embeddings_output}")

    return results, db_meta, query_meta


def run_task2_on_queries(args, task1_results, db_meta, device, extractor, matcher):
    """Run Task 2 angle estimation on all queries."""

    print("\nTASK 2: LightGlue Angle Estimation\n")

    task2_results = {}

    # Limit queries for testing if specified
    items = list(task1_results.items())
    if args.max_queries:
        items = items[:args.max_queries]
        print(f"Processing only first {args.max_queries} queries (test mode)")

    for qid, result in tqdm(items, desc="Processing queries"):
        qid_int = int(qid)

        # Get query image path
        query_path = Path(args.image_dir) / f"{qid_int}.jpg"
        if not query_path.exists():
            print(f"WARNING: Query image not found: {query_path}")
            task2_results[qid_int] = {
                "query_id": qid_int,
                "error": "query_image_not_found",
                "estimated_angle": None,
                "consistency_error": None,
                "match_results": [],
            }
            continue

        # Get candidate images from Task 1 top-k
        topk_items = result.get("topk", [])
        candidate_paths = []
        candidate_angles = []

        for item in topk_items:
            cand_id = int(item["id"])
            cand_path = Path(args.image_dir) / f"{cand_id}.jpg"
            if cand_path.exists():
                candidate_paths.append(str(cand_path))
                # Get ground truth angle if available
                if cand_id in db_meta:
                    candidate_angles.append(db_meta[cand_id].get("angle"))
                else:
                    candidate_angles.append(None)

        # Run Task 2 angle estimation
        estimated_angle, consistency_error, match_results, avg_matches_used = estimate_query_orientation(
            query_path=str(query_path),
            candidate_paths=candidate_paths,
            candidate_angles=candidate_angles,
            device=device,
            extractor=extractor,
            matcher=matcher,
            min_matches=args.min_matches,
            focal_length_scale=args.focal_length_scale,
            save_visualizations=args.save_visualizations,
            output_dir=args.output_dir,
            query_id=str(qid_int),
        )

        task2_results[qid_int] = {
            "query_id": qid_int,
            "estimated_angle": estimated_angle,
            "consistency_error": consistency_error,
            "avg_matches_used": avg_matches_used,
            "match_results": match_results,
        }

    # Save Task 2 results
    task2_output = Path(args.output_dir) / f"task2_results_{args.city}.json"
    save_json(str(task2_output), task2_results)
    print(f"Saved Task 2 results to {task2_output}")

    return task2_results


def run_user_mode(args, device, extractor, matcher):
    """Run pipeline on user images against city database."""

    print("\nUSER MODE: Processing images against database\n")

    # Load user images
    image_dir = Path(args.user_image_dir)
    user_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not user_images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(user_images)} user images")

    # Load MegaLoc model for Task 1
    print("\nLoading MegaLoc model for Task 1 retrieval...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Load database embeddings from saved file
    embeddings_path = Path(args.output_dir) / "embeddings.pt"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Database embeddings not found at {embeddings_path}. "
            "Please run the full pipeline first: python pipeline.py --topk 100"
        )
    
    print(f"\nLoading database embeddings from {embeddings_path}...")
    checkpoint = torch.load(str(embeddings_path), map_location=device)
    db_ids = checkpoint["database"]["ids"]
    db_emb = checkpoint["database"]["embeddings"].to(device)
    db_meta = checkpoint.get("db_meta", {})
    print(f"Loaded {len(db_ids)} database embeddings")

    # Process each user image independently
    results = {}

    for query_img in tqdm(user_images, desc="Processing user images"):
        query_id = query_img.stem

        # Compute query embedding
        q_ids, q_emb = compute_embeddings(
            model=model,
            device=device,
            annotations_path=None,
            image_dir=str(query_img.parent),
            batch_size=1,
            num_workers=0,
            custom_images=[str(query_img)],
        )

        # Run Task 1: Retrieve top-k from database
        task1_result = retrieve_topk(
            db_ids=db_ids,
            db_emb=db_emb,
            q_ids=q_ids,
            q_emb=q_emb.to(device),
            db_meta=db_meta,
            topk=args.topk,
            estimate_topk=args.position_estimation_topk,
        )

        # Get top-k candidates for Task 2
        topk_items = task1_result.get(q_ids[0], {}).get("topk", [])
        position_estimate = task1_result.get(q_ids[0], {}).get("position_estimates", {}).get("rank_medoid_top5")

        # Prepare candidates for Task 2
        candidate_paths = []
        candidate_angles = []
        for item in topk_items:
            cand_id = int(item["id"])
            img_path = str(Path(args.image_dir) / f"{cand_id}.jpg")
            candidate_paths.append(img_path)
            candidate_angles.append(db_meta.get(cand_id, {}).get("angle"))

        # Run Task 2: Estimate angle
        estimated_angle, consistency_error, match_results, avg_matches_used = estimate_query_orientation(
            query_path=str(query_img),
            candidate_paths=candidate_paths,
            candidate_angles=candidate_angles,
            device=device,
            extractor=extractor,
            matcher=matcher,
            min_matches=args.min_matches,
            focal_length_scale=args.focal_length_scale,
            save_visualizations=args.save_visualizations,
            output_dir=args.output_dir,
            query_id=query_id,
        )

        results[query_id] = {
            "query_id": query_id,
            "query_path": str(query_img),
            "position_estimate": position_estimate,
            "estimated_angle": estimated_angle,
            "consistency_error": consistency_error,
            "avg_matches_used": avg_matches_used,
            "match_results": match_results,
            "topk": topk_items,
        }

    return results


def generate_yaml_report(task1_results, task2_results, db_meta, query_meta, args):
    """Generate YAML evaluation report."""

    print("\nGenerating YAML Evaluation Report")

    report = {
        "pipeline_config": {
            "task1_topk": args.topk,
            "task2_min_matches": args.min_matches,
            "user_mode": args.user_images,
            "city": args.city,
        },
        "summary": {},
        "per_query_results": {},
    }

    # Compute summary statistics
    total_queries = len(task1_results)
    queries_with_angles = sum(1 for r in task2_results.values() if r.get("estimated_angle") is not None)
    avg_consistency = np.mean([r.get("consistency_error", 0) for r in task2_results.values() if r.get("consistency_error") is not None]) if task2_results else 0

    report["summary"] = {
        "total_queries": total_queries,
    }

    # Compute Task 1 metrics if ground truth available
    if query_meta and not args.user_images:
        # Calculate position errors
        position_errors = []
        for qid, result in task1_results.items():
            qid_int = int(qid)
            # Check both int and string keys
            meta_key = qid_int if qid_int in query_meta else (qid if qid in query_meta else None)
            if meta_key is not None:
                gt_lat = query_meta[meta_key]["lat"]
                gt_lon = query_meta[meta_key]["lon"]
                est = result.get("position_estimates", {}).get("rank_medoid_top5")
                if est:
                    err = haversine_m(gt_lat, gt_lon, est["lat"], est["lon"])
                    position_errors.append(err)

        if position_errors:
            report["summary"]["task1_position_error_m"] = {
                "mean": float(np.mean(position_errors)),
                "median": float(np.median(position_errors)),
            }

        # Compute Task 1 Recall@K (50m threshold)
        try:
            topk_for_eval = {qid: result["topk"] for qid, result in task1_results.items()}
            recall_metrics, _ = evaluate_task1(
                topk_results=topk_for_eval,
                db_meta=db_meta,
                query_meta=query_meta,
                ks=[1, 5, 10],
                distance_thresh_m=50,
                angle_thresh_deg=None,
            )
            report["summary"]["task1_recall"] = {
                "@1_50m": recall_metrics.get("recall@1", 0),
                "@5_50m": recall_metrics.get("recall@5", 0),
                "@10_50m": recall_metrics.get("recall@10", 0),
            }
        except Exception as e:
            print(f"Warning: Could not compute Task 1 Recall@K: {e}")

    # Compute angle errors if ground truth available
    if query_meta and not args.user_images:
        angle_errors = []
        for qid, t2_result in task2_results.items():
            qid_int = int(qid)
            # Check both int and string keys
            meta_key = qid_int if qid_int in query_meta else (qid if qid in query_meta else None)
            if meta_key is not None and t2_result.get("estimated_angle") is not None:
                gt_angle = query_meta[meta_key]["angle"]
                est_angle = t2_result["estimated_angle"]
                err = angle_diff_deg(gt_angle, est_angle)
                angle_errors.append(err)

        if angle_errors:
            angle_errors_array = np.array(angle_errors)
            report["summary"]["task2_angle_error_deg"] = {
                "mean": float(np.mean(angle_errors)),
                "median": float(np.median(angle_errors)),
                "consistency_error": float(avg_consistency),
            }
            report["summary"]["task2_angle_success"] = {
                "threshold_10deg": float(np.mean(angle_errors_array <= 10.0)),
                "threshold_20deg": float(np.mean(angle_errors_array <= 20.0)),
                "threshold_30deg": float(np.mean(angle_errors_array <= 30.0)),
            }

    # Per-query results
    for qid in task1_results.keys():
        qid_int = int(qid)
        t1_result = task1_results.get(qid, {})
        t2_result = task2_results.get(qid_int, {})

        query_report = {
            "task1": {
                "position_estimate": t1_result.get("position_estimates", {}).get("rank_medoid_top5"),
            },
            "task2": {
                "estimated_angle": t2_result.get("estimated_angle"),
                "consistency_error": t2_result.get("consistency_error"),
                "avg_matches_used": t2_result.get("avg_matches_used"),
            },
        }

        # Add ground truth if available (check both int and string keys)
        meta_key = qid_int if qid_int in query_meta else (qid if qid in query_meta else None)
        if query_meta and meta_key is not None:
            query_report["ground_truth"] = {
                "lat": query_meta[meta_key]["lat"],
                "lon": query_meta[meta_key]["lon"],
                "angle": query_meta[meta_key]["angle"],
            }

            # Add errors
            if t1_result.get("position_estimates", {}).get("rank_medoid_top5"):
                est = t1_result["position_estimates"]["rank_medoid_top5"]
                query_report["errors"] = {
                    "position_error_m": haversine_m(
                        query_meta[meta_key]["lat"], query_meta[meta_key]["lon"],
                        est["lat"], est["lon"]
                    ),
                }
            if t2_result.get("estimated_angle") is not None:
                err = angle_diff_deg(query_meta[meta_key]["angle"], t2_result["estimated_angle"])
                if "errors" not in query_report:
                    query_report["errors"] = {}
                query_report["errors"]["angle_error_deg"] = err

        report["per_query_results"][qid] = query_report

    # Save YAML (with city name in filename)
    yaml_output = Path(args.output_dir) / (args.yaml_output or f"pipeline_evaluation_{args.city}.yaml")
    yaml_output.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_output, "w") as f:
        yaml.dump(convert_numpy_types(report), f, default_flow_style=False, sort_keys=False)

    print(f"Saved YAML evaluation report to {yaml_output}")
    print("\nSummary:")
    print(yaml.dump(convert_numpy_types(report["summary"]), default_flow_style=False))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Combined pipeline: Task 1 (MegaLoc retrieval) + Task 2 (LightGlue angle estimation)"
    )

    # Input/Output
    parser.add_argument("--image-dir", default=config.IMAGE_DIR, help="Directory containing images")
    parser.add_argument("--annotations-db", default=config.TASK1_ANNOTATIONS_DB, help="Database annotations")
    parser.add_argument("--annotations-query", default=config.TASK1_ANNOTATIONS_QUERY, help="Query annotations")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--yaml-output", default="pipeline_evaluation.yaml", help="YAML output filename")

    # Task 1 options
    parser.add_argument("--topk", type=int, default=config.PIPELINE_TOPK, help="Number of top-k retrievals (recommend 100+ for Task 2)")
    parser.add_argument("--position-estimation-topk", type=int, default=config.PIPELINE_POSITION_ESTIMATION_TOPK, help="Top-k for position estimation")
    parser.add_argument("--checkpoint", default=config.TASK1_CHECKPOINT, help="Model checkpoint")
    parser.add_argument("--multi-gpu", action="store_true", help="Use DataParallel")
    parser.add_argument("--batch-size", type=int, default=config.TASK1_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.TASK1_NUM_WORKERS)

    # Task 2 options
    parser.add_argument("--min-matches", type=int, default=config.PIPELINE_TASK2_MIN_MATCHES, help="Minimum LightGlue matches for angle estimation")
    parser.add_argument("--focal-length-scale", type=float, default=config.PIPELINE_TASK2_FOCAL_LENGTH_SCALE, help="Focal length scale factor")
    parser.add_argument("--save-visualizations", action="store_true", default=config.PIPELINE_TASK2_SAVE_VISUALIZATIONS, help="Save LightGlue match visualizations")
    parser.add_argument("--max-queries", type=int, default=config.PIPELINE_MAX_QUERIES, help="Max queries to process (for testing, default: all)")

    # City selection (for report naming)
    parser.add_argument("--city", choices=["brussels", "liege"], default=config.PIPELINE_CITY, help="City name for report filenames (default: brussels)")

    # User mode
    parser.add_argument("--user-images", action="store_true", default=config.PIPELINE_USER_IMAGES, help="Run on user images without metadata")
    parser.add_argument("--user-image-dir", default=config.PIPELINE_USER_IMAGE_DIR, help="Directory with user images")

    # Control flow
    parser.add_argument("--skip-task1", action="store_true", default=config.PIPELINE_SKIP_TASK1, help="Skip Task 1 (use existing results)")
    parser.add_argument("--task1-results", default=config.PIPELINE_TASK1_RESULTS, help="Path to existing Task 1 results JSON")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.user_images:
        # User mode: images without metadata
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

        results = run_user_mode(args, device, extractor, matcher)

        # Save results
        output_path = Path(args.output_dir) / "user_results.json"
        save_json(str(output_path), results)
        print(f"Saved user results to {output_path}")

        # Generate simplified YAML report
        simplified_results = {}
        for img_id, data in results.items():
            simplified_results[img_id] = {
                "position_estimate": data.get("position_estimate"),
                "estimated_angle": data.get("estimated_angle"),
                "consistency_error": data.get("consistency_error"),
                "avg_matches_used": data.get("avg_matches_used"),
            }
        report = {
            "mode": "user_images",
            "total_images": len(results),
            "results": simplified_results,
        }
        yaml_output = Path(args.output_dir) / "user_results_summary.yaml"
        with open(yaml_output, "w") as f:
            yaml.dump(convert_numpy_types(report), f, default_flow_style=False, sort_keys=False)
        print(f"Saved YAML report to {yaml_output}")

    else:
        # Full pipeline mode with metadata

        # Task 1: Retrieval
        if args.skip_task1 and args.task1_results:
            print(f"Loading existing Task 1 results from {args.task1_results}")
            with open(args.task1_results, "r") as f:
                task1_results = json.load(f)
            db_meta, query_meta = load_embeddings_metadata(
                str(Path(args.output_dir) / "embeddings.pt")
            )
        else:
            task1_results, db_meta, query_meta = run_task1(args, device)

        # Task 2: Angle Estimation
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

        task2_results = run_task2_on_queries(
            args, task1_results, db_meta, device, extractor, matcher
        )

        # Generate YAML evaluation report
        generate_yaml_report(task1_results, task2_results, db_meta, query_meta, args)

if __name__ == "__main__":
    main()
