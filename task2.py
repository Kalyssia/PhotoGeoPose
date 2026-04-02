import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from lightglue import LightGlue, SuperPoint, utils, viz2d
from scipy.spatial.transform import Rotation


def angle_diff(a, b) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)

def extract_image_id(path: str) -> int:
    name = os.path.basename(path)
    id, _ = os.path.splitext(name)
    return int(id)

def real_angle_change(img0: str, img1: str, metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    by_id = {int(item["id"]): float(item["angle"]) for item in metadata}
    id0 = extract_image_id(img0)
    id1 = extract_image_id(img1)

    if id0 not in by_id or id1 not in by_id:
        raise KeyError(
            f"Image ids {id0} and/or {id1} not found in {metadata_path}."
        )
    a0 = by_id[id0]
    a1 = by_id[id1]
    return a0, a1, angle_diff(a0, a1)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

# Load Images
img0_path = 'images/1.png'
img1_path = 'images/2.png'
metadata_path = 'images/metadata.json'
image0 = utils.load_image(img0_path).to(device)
image1 = utils.load_image(img1_path).to(device)

# Use SyperPoint to extract features and LightGlue to match them
features0 = extractor.extract(image0)
features1 = extractor.extract(image1)
matches = matcher({'image0': features0, 'image1': features1})
kpts0, kpts1 = features0['keypoints'], features1['keypoints']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# Visualization of matches
# features0, features1, matches = [utils.rbd(x) for x in [features0, features1, matches]]
# matches = matches['matches']
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
# viz2d.add_text(0, f'Stop 1: {len(m_kpts0)} matches found', fs=20)
# plt.show()

# Convert to numpy for OpenCV
pts0 = m_kpts0.cpu().numpy()
pts1 = m_kpts1.cpu().numpy()

# Camera parameters
h, w = cv2.imread(img0_path).shape[:2]
focal_length = 1.0 * w
K = np.array([[focal_length, 0, w / 2],
              [0, focal_length, h / 2],
              [0, 0, 1]])

# Recover Pose using Essential Matrix
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K)

# Extract yaw angle from rotation matrix ('yxz' gives yaw (Y), pitch (X), roll (Z))
r = Rotation.from_matrix(R)
angles = r.as_euler('yxz', degrees=True)
yaw = angles[0]

# Comparison with real angle value from metadata
real_a0, real_a1, real_delta = real_angle_change(img0_path, img1_path, metadata_path)
comparison_error = angle_diff(yaw, real_delta)

print(f"Estimated angle: {yaw:.2f} degrees")
print(f"Real angle: {real_delta:.2f} degrees")
print(f"Angle difference: {comparison_error:.2f} degrees")