import cv2
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, utils, viz2d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# If you want to test, you can change those
REF_IMAGE_PATH = "images/170533595024759.jpg"
IMAGE_DIR = "images"
MIN_MATCHES = 500 # This is particularly influencing the result
FEW_MATCHES_THRESHOLD = 50 # Images with fewer than this number of matches are considered "few matches"
FOCAL_LENGTH_SCALE = 1 # Adjust if the focal length in pixels is different from the image width (values around 1.0 seem to work better)
OUTPUT_MANY_MATCHES_DIR = "images/output/many_matches"
OUTPUT_FEW_MATCHES_DIR = "images/output/few_matches"

## Data handling functions

# Extracts the image ID from the filename
def extract_image_id(path):
    name = os.path.basename(path)
    id, _ = os.path.splitext(name)
    return int(id)

# Loads metadata and returns an ID and an angle (we do not use the position for this task)
def get_metadata_angles(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {int(item["id"]): float(item["angle"]) for item in metadata}

# Lists candidate images in the directory, excluding the reference image.
def get_candidate_images(images_dir, reference_path):
    reference_name = os.path.basename(reference_path)
    candidates = []
    for name in os.listdir(images_dir):
        if name == reference_name:
            continue
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            candidates.append(os.path.join(images_dir, name))
    return candidates

## Angle calculation functions

# Computes the smallest difference between two angles (in degrees), accounting for wrap-around at 360 degrees.
def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

# Normalizes an angle (in degrees) to the range [0, 360) degrees.
def normalize_angle(angle):
    return angle % 360

# Computes the circular mean of a list of angles (in degrees), accounting for wrap-around.
def circular_mean_deg(angles):
    radians = np.deg2rad(angles)
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()
    return normalize_angle(np.rad2deg(np.arctan2(sin_mean, cos_mean)))

# Computes the mean circular error between a reference angle and a list of angles (in degrees).
def mean_circular_error(reference, angles):
    return float(np.mean([angle_diff(reference, angle) for angle in angles]))

# Runs LightGlue matching and returns matched keypoint arrays (as numpy) and match count.
def compute_matches(ref_features, other_features, matcher):
    raw = matcher({"image0": ref_features, "image1": other_features})
    feat0, feat1, raw = [utils.rbd(x) for x in [ref_features, other_features, raw]]
    matches = raw["matches"]
    match_count = len(matches)
    if match_count == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    pts0 = feat0["keypoints"][matches[..., 0]].cpu().numpy()
    pts1 = feat1["keypoints"][matches[..., 1]].cpu().numpy()
    return pts0, pts1, match_count

# Estimates the relative yaw angle (in degrees) from already-matched point pairs and K.
def estimate_relative_yaw(pts0, pts1, K):
    # Use RANSAC to find the essential matrix
    E, _ = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC)
    # Recover the relative pose (rotation) from the essential matrix
    _, R, _, _ = cv2.recoverPose(E, pts0, pts1, K)
    # Extract the yaw angle from the rotation matrix. The "yxz" order corresponds to yaw-pitch-roll and we take the first angle which is the yaw
    yaw = Rotation.from_matrix(R).as_euler("yxz", degrees=True)[0]
    return yaw

# Saves a LightGlue match plot using viz2d (two images + line correspondences).
def save_match_visualization(image0, image1, m_kpts0, m_kpts1, match_count, save_path):
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

# Main function to estimate the reference image orientation
def estimate_reference_orientation(reference_path, images_dir, metadata_path):
    metadata_angles = get_metadata_angles(metadata_path)
    candidate_images = get_candidate_images(images_dir, reference_path)
    ref_image = cv2.imread(reference_path)

    # Camera intrinsic matrix K estimation
    h, w = ref_image.shape[:2]
    focal_length = FOCAL_LENGTH_SCALE * w
    K = np.array(
        [[focal_length, 0, w / 2], 
         [0, focal_length, h / 2],
         [0, 0, 1]]
    )

    # Extract features from the reference image once, to reuse for all comparisons
    ref_image_tensor = utils.load_image(reference_path).to(device)
    ref_features = extractor.extract(ref_image_tensor)
    ref_viz_image = utils.load_image(reference_path)

    results = []
    many_matches_viz = []  # (pts0, pts1, match_count, other_path) for images with >= MIN_MATCHES matches
    few_matches_viz = []   # (pts0, pts1, match_count, other_path) for images with < FEW_MATCHES_THRESHOLD matches

    # Main loop to calculate the relative yaw for each candidate image and estimate the reference orientation
    for other_path in tqdm(candidate_images, desc="Calculating reference image orientation"):
        id = extract_image_id(other_path)
        if id not in metadata_angles:
            continue

        # Extract features from the other image
        other_image_tensor = utils.load_image(other_path).to(device)
        other_features = extractor.extract(other_image_tensor)

        # Compute matches (always, so we can visualize regardless of count)
        pts0, pts1, match_count = compute_matches(ref_features, other_features, matcher)

        # Categorize by match count and store data for visualization
        if match_count >= MIN_MATCHES:
            many_matches_viz.append((pts0, pts1, match_count, other_path))
        elif match_count < FEW_MATCHES_THRESHOLD and match_count > 10:  # Only consider it "few matches" if there is at least 10 matches to visualize
            few_matches_viz.append((pts0, pts1, match_count, other_path))

        if match_count < MIN_MATCHES:
            continue

        yaw = estimate_relative_yaw(pts0, pts1, K)
        other_angle = metadata_angles[id]
        calculated_angle = normalize_angle(other_angle + yaw)
        results.append(
            {
                "image": os.path.basename(other_path),
                "matches": match_count,
                "other_angle": other_angle,
                "relative_yaw": yaw,
                "calculated_angle": calculated_angle,
            }
        )

    # Save match visualizations to output directories
    os.makedirs(OUTPUT_MANY_MATCHES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FEW_MATCHES_DIR, exist_ok=True)
    for pts0, pts1, match_count, other_path in many_matches_viz:
        other_img_tensor = utils.load_image(other_path)
        out_name = os.path.splitext(os.path.basename(other_path))[0] + "_matches.jpg"
        save_match_visualization(ref_viz_image, other_img_tensor, pts0, pts1, match_count, os.path.join(OUTPUT_MANY_MATCHES_DIR, out_name))
    for pts0, pts1, match_count, other_path in few_matches_viz:
        other_img_tensor = utils.load_image(other_path)
        out_name = os.path.splitext(os.path.basename(other_path))[0] + "_matches.jpg"
        save_match_visualization(ref_viz_image, other_img_tensor, pts0, pts1, match_count, os.path.join(OUTPUT_FEW_MATCHES_DIR, out_name))

    print(f"\nSaved {len(many_matches_viz)} match visualizations (>= {MIN_MATCHES} matches) to '{OUTPUT_MANY_MATCHES_DIR}'")
    print(f"Saved {len(few_matches_viz)} match visualizations (< {FEW_MATCHES_THRESHOLD} matches) to '{OUTPUT_FEW_MATCHES_DIR}'\n")

    estimated_angle = circular_mean_deg([result["calculated_angle"] for result in results])
    error = mean_circular_error(estimated_angle, [result["calculated_angle"] for result in results])

    return reference_path, estimated_angle, error, results


## Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

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
    error_vs_true = angle_diff(estimated_orientation, true_angle)
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