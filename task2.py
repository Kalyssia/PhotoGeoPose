import cv2
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, utils, viz2d
from scipy.spatial.transform import Rotation

from utils import angle_diff_deg, normalize_angle, circular_mean_deg, mean_circular_error
import config

# Task 2 configuration (can be overridden via config.py)
REF_IMAGE_PATH = f"{config.IMAGE_DIR}/{config.TASK2_REF_IMAGE_ID}.jpg"
IMAGE_DIR = config.IMAGE_DIR
MIN_MATCHES = config.TASK2_MIN_MATCHES
FEW_MATCHES_THRESHOLD = config.TASK2_FEW_MATCHES_THRESHOLD
FOCAL_LENGTH_SCALE = config.TASK2_FOCAL_LENGTH_SCALE
OUTPUT_MANY_MATCHES_DIR = config.TASK2_OUTPUT_MANY_MATCHES_DIR
OUTPUT_FEW_MATCHES_DIR = config.TASK2_OUTPUT_FEW_MATCHES_DIR

## Data handling functions

def extract_image_id(path):
    """Extracts the image ID from the filename."""
    name = os.path.basename(path)
    id, _ = os.path.splitext(name)
    return int(id)

def get_metadata_angles(metadata_path):
    """Loads metadata and returns an ID and an angle (we do not use the position for this task)."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {int(item["id"]): float(item["angle"]) for item in metadata}

def get_candidate_images(images_dir, reference_path):
    """Lists candidate images in the directory, excluding the reference image."""
    reference_name = os.path.basename(reference_path)
    candidates = []
    for name in os.listdir(images_dir):
        if name == reference_name:
            continue
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            candidates.append(os.path.join(images_dir, name))
    return candidates

## Angle calculation functions are imported from utils.py

def compute_matches(ref_features, other_features, matcher):
    """Runs LightGlue matching and returns matched keypoint arrays (as numpy) and match count."""
    raw = matcher({"image0": ref_features, "image1": other_features})
    feat0, feat1, raw = [utils.rbd(x) for x in [ref_features, other_features, raw]]
    matches = raw["matches"]
    match_count = len(matches)
    if match_count == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    pts0 = feat0["keypoints"][matches[..., 0]].cpu().numpy()
    pts1 = feat1["keypoints"][matches[..., 1]].cpu().numpy()
    return pts0, pts1, match_count

def estimate_relative_yaw(pts0, pts1, K):
    """Estimates the relative yaw angle (in degrees) from matched point pairs."""
    if len(pts0) < 5:
        return None
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, _, _ = cv2.recoverPose(E, pts0, pts1, K)
    yaw = Rotation.from_matrix(R).as_euler("yxz", degrees=True)[0]
    return yaw

def save_match_visualization(image0, image1, m_kpts0, m_kpts1, match_count, save_path):
    """Saves a LightGlue match plot using viz2d (two images + line correspondences)."""
    if isinstance(image0, torch.Tensor):
        image0 = image0.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(image1, torch.Tensor):
        image1 = image1.detach().cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(14, 7))
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f"{match_count} matches", fs=18)
    fig = plt.gcf()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def estimate_query_orientation(
    query_path,
    candidate_paths,
    candidate_angles,
    device,
    extractor,
    matcher,
    min_matches=500,
    focal_length_scale=1.0,
    save_visualizations=False,
    output_dir=None,
    query_id=None,
):
    """Estimate query orientation using LightGlue matches against candidates."""
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"WARNING: Could not load query image: {query_path}")
        return None, None, [], 0

    h, w = query_img.shape[:2]
    focal_length = focal_length_scale * w
    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])

    # Extract features from query image once
    try:
        query_tensor = utils.load_image(query_path).to(device)
        query_features = extractor.extract(query_tensor)
        query_viz_image = utils.load_image(query_path) if save_visualizations else None
    except Exception as e:
        print(f"WARNING: Failed to extract features from query: {e}")
        return None, None, [], 0

    results = []
    used_candidates = []  # Candidates used for angle estimation
    visualization_data = []  # Store for saving visualizations

    for cand_path, cand_angle in zip(candidate_paths, candidate_angles):
        if not os.path.exists(cand_path):
            continue

        try:
            cand_tensor = utils.load_image(cand_path).to(device)
            cand_features = extractor.extract(cand_tensor)
        except Exception:
            continue

        pts0, pts1, match_count = compute_matches(query_features, cand_features, matcher)

        result = {
            "candidate_image": os.path.basename(cand_path),
            "matches": match_count,
            "used": False,
        }

        if match_count >= min_matches:
            yaw = estimate_relative_yaw(pts0, pts1, K)
            if yaw is not None and cand_angle is not None:
                calculated_angle = normalize_angle(cand_angle + yaw)
                result["relative_yaw"] = yaw
                result["calculated_angle"] = calculated_angle
                result["used"] = True
                results.append(result)
                used_candidates.append({
                    "path": cand_path,
                    "id": int(os.path.splitext(os.path.basename(cand_path))[0]),
                    "matches": match_count,
                })
                # Store visualization data
                if save_visualizations and query_viz_image is not None:
                    visualization_data.append((pts0, pts1, match_count, cand_path, cand_tensor))
            elif yaw is not None:
                # No ground truth angle available (user mode)
                result["relative_yaw"] = yaw
                result["used"] = True
                results.append(result)
                used_candidates.append({
                    "path": cand_path,
                    "id": int(os.path.splitext(os.path.basename(cand_path))[0]),
                    "matches": match_count,
                })
                if save_visualizations and query_viz_image is not None:
                    visualization_data.append((pts0, pts1, match_count, cand_path, cand_tensor))
        else:
            results.append(result)

    # Save visualizations if requested
    if save_visualizations and output_dir and visualization_data:
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        for pts0, pts1, match_count, cand_path, cand_tensor in visualization_data:
            out_name = f"{query_id}_{os.path.splitext(os.path.basename(cand_path))[0]}_matches.jpg"
            save_path = os.path.join(viz_dir, out_name)
            save_match_visualization(query_viz_image, cand_tensor, pts0, pts1, match_count, save_path)
        print(f"Saved {len(visualization_data)} visualizations to {viz_dir}")

    if not results:
        return None, None, [], 0

    used_angles = [r["calculated_angle"] for r in results if r.get("calculated_angle") is not None]

    if not used_angles:
        return None, None, results, 0

    # Calculate average matches of candidates used for angle estimation
    avg_matches_used = sum(c["matches"] for c in used_candidates) / len(used_candidates) if used_candidates else 0

    estimated_angle = circular_mean_deg(used_angles)
    consistency_error = mean_circular_error(estimated_angle, used_angles)

    return estimated_angle, consistency_error, results, avg_matches_used

def estimate_reference_orientation(reference_path, images_dir, metadata_path):
    """Estimate reference image orientation using the reusable estimate_query_orientation function."""
    metadata_angles = get_metadata_angles(metadata_path)
    candidate_images = get_candidate_images(images_dir, reference_path)

    # Build candidate_paths and candidate_angles lists
    candidate_paths = []
    candidate_angles = []
    for cand_path in candidate_images:
        cand_id = extract_image_id(cand_path)
        if cand_id in metadata_angles:
            candidate_paths.append(cand_path)
            candidate_angles.append(metadata_angles[cand_id])

    # Use the reusable function with visualization enabled
    ref_id = extract_image_id(reference_path)
    estimated_angle, consistency_error, results, _ = estimate_query_orientation(
        query_path=reference_path,
        candidate_paths=candidate_paths,
        candidate_angles=candidate_angles,
        device=device,
        extractor=extractor,
        matcher=matcher,
        min_matches=MIN_MATCHES,
        focal_length_scale=FOCAL_LENGTH_SCALE,
        save_visualizations=True,
        output_dir=os.path.dirname(OUTPUT_MANY_MATCHES_DIR),
        query_id=str(ref_id),
    )

    return reference_path, estimated_angle, consistency_error, results


## Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=config.TASK2_MAX_NUM_KEYPOINTS).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

if __name__ == "__main__":
    # Run the estimation
    metadata_path = os.path.join(IMAGE_DIR, "metadata.json")
    reference_path, estimated_orientation, consistency_error, results = estimate_reference_orientation(REF_IMAGE_PATH, IMAGE_DIR, metadata_path)

    # Print results
    print(f"Reference image: {os.path.basename(reference_path)}")
    print(f"Estimated angle: {estimated_orientation:.2f} degrees")
    print(f"Consistency error: {consistency_error:.2f} degrees\n")

    # Compare with ground-truth if available
    metadata_angles = get_metadata_angles(metadata_path)
    ref_id = extract_image_id(reference_path)
    if ref_id in metadata_angles:
        true_angle = metadata_angles[ref_id]
        error_vs_true = angle_diff_deg(estimated_orientation, true_angle)
        print(f"Ground-truth reference angle: {true_angle:.2f} degrees")
        print(f"Error vs ground-truth: {error_vs_true:.2f} degrees\n")

    for result in results:
        print(
            f"{result['image']}: "
            f"matches = {result['matches']}, "
            # f"other image angle={result['other_angle']:.2f}, "
            f"calculated yaw btw images = {result['relative_yaw']:.2f}, "
            f"calculated angle = {result['calculated_angle']:.2f}"
        )