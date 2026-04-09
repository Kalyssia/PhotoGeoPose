import cv2
import numpy as np
import torch
import json
import os
from lightglue import LightGlue, SuperPoint, utils
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# If you want to test, you can change those
REF_IMAGE_PATH = "images/126774469409595.jpg"
IMAGE_DIR = "images"
MIN_MATCHES = 500 # This is particularly influencing the result
FOCAL_LENGTH_SCALE = 1 # Adjust if the focal length in pixels is different from the image width (values around 1.0 seem to work better)

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

# Estimates the relative yaw angle (in degrees) between the reference image and another image using LigthGlue matches and the essential matrix.
# Returns the estimated yaw and the number of matches used for the estimation. If not enough matches are found, returns None and the match count.
def estimate_relative_yaw(ref_features, other_features, matcher, K):
    matches = matcher({"image0": ref_features, "image1": other_features})
    ref_features, other_features, matches = [utils.rbd(x) for x in [ref_features, other_features, matches]]
    matches = matches["matches"]

    if len(matches) < MIN_MATCHES:
        return None, len(matches)

    kpts0, kpts1 = ref_features["keypoints"], other_features["keypoints"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    pts0 = m_kpts0.cpu().numpy()
    pts1 = m_kpts1.cpu().numpy()

    # Use RANSAC to find the essential matrix
    E, _ = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC)
    # Recover the relative pose (rotation) from the essential matrix
    _, R, _, _ = cv2.recoverPose(E, pts0, pts1, K)
    # Extract the yaw angle from the rotation matrix. The "yxz" order corresponds to yaw-pitch-roll and we take the first angle which is the yaw
    yaw = Rotation.from_matrix(R).as_euler("yxz", degrees=True)[0]
    return yaw, len(matches)

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

    results = []
    # Main loop to calculate the relative yaw for each candidate image and estimate the reference orientation
    for other_path in tqdm(candidate_images, desc="Calculating reference image orientation"):
        id = extract_image_id(other_path)
        if id not in metadata_angles:
            continue

        # Extract features from the other image
        other_image_tensor = utils.load_image(other_path).to(device)
        other_features = extractor.extract(other_image_tensor)

        # Estimate the relative yaw to the reference image
        yaw, match_count = estimate_relative_yaw(ref_features, other_features, matcher, K)
        if yaw is None:
            continue

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