#!/usr/bin/env python3
"""
Process pose joint data for BOTH fencers (left + right) from v3 extractor.
- Reads *_pose.json that contain {"frame": n, "keypoints": [left, right]} per frame.
- Writes separate outputs for left, right, and a flipped-right-as-left view.
- Preserves subfolder structure from the overlay root in each output root.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import logging
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent
DEFAULT_POSE_OVERLAY_DIR = HERE / "working" / "pose_overlay_v3"
DEFAULT_OUTPUT_ROOT = HERE / "working" / "processed_joints_v3"

BODY_KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

HEAD_JOINTS = 5  # first 5 keypoints are head

def compute_angles(keypoints):
    """Compute joint angles (elbow and knee angles)."""
    # Keypoint indices after removing head keypoints
    idx = {
        "left_shoulder": 0, "right_shoulder": 1, "left_elbow": 2, "right_elbow": 3,
        "left_wrist": 4, "right_wrist": 5, "left_hip": 6, "right_hip": 7,
        "left_knee": 8, "right_knee": 9, "left_ankle": 10, "right_ankle": 11
    }
    
    def angle_between_vectors(a, b, c):
        """Calculate angle at point b between vectors ba and bc."""
        # Handle NaN values
        mask = ~(np.isnan(a).any(axis=-1) | np.isnan(b).any(axis=-1) | np.isnan(c).any(axis=-1))
        
        ba = a - b
        bc = c - b
        
        # Calculate dot product and magnitudes
        dot_product = np.sum(ba * bc, axis=-1)
        magnitude_ba = np.sqrt(np.sum(ba * ba, axis=-1))
        magnitude_bc = np.sqrt(np.sum(bc * bc, axis=-1))
        
        # Avoid division by zero
        magnitude_product = magnitude_ba * magnitude_bc
        magnitude_product = np.where(magnitude_product == 0, 1e-8, magnitude_product)
        
        # Calculate cosine and angle
        cos_angle = dot_product / magnitude_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.arccos(cos_angle)
        
        # Set NaN for invalid frames
        angles = np.where(mask, angles, np.nan)
        
        return angles

    angles = {}
    
    # Left elbow angle: shoulder -> elbow -> wrist
    angles['left_elbow'] = angle_between_vectors(
        keypoints[:, idx["left_shoulder"]], 
        keypoints[:, idx["left_elbow"]], 
        keypoints[:, idx["left_wrist"]]
    )
    
    # Right elbow angle: shoulder -> elbow -> wrist
    angles['right_elbow'] = angle_between_vectors(
        keypoints[:, idx["right_shoulder"]], 
        keypoints[:, idx["right_elbow"]], 
        keypoints[:, idx["right_wrist"]]
    )
    
    # Left knee angle: hip -> knee -> ankle
    angles['left_knee'] = angle_between_vectors(
        keypoints[:, idx["left_hip"]], 
        keypoints[:, idx["left_knee"]], 
        keypoints[:, idx["left_ankle"]]
    )
    
    # Right knee angle: hip -> knee -> ankle
    angles['right_knee'] = angle_between_vectors(
        keypoints[:, idx["right_hip"]], 
        keypoints[:, idx["right_knee"]], 
        keypoints[:, idx["right_ankle"]]
    )
    
    return angles

def compute_distances(keypoints):
    """Compute distances between key body parts."""
    idx = {
        "left_shoulder": 0, "right_shoulder": 1, "left_elbow": 2, "right_elbow": 3,
        "left_wrist": 4, "right_wrist": 5, "left_hip": 6, "right_hip": 7,
        "left_knee": 8, "right_knee": 9, "left_ankle": 10, "right_ankle": 11
    }
    
    def distance(a, b):
        diff = a - b
        return np.sqrt(np.sum(diff * diff, axis=-1))
    
    distances = {}
    
    # Distance between hands (useful for fencing)
    distances['hand_to_hand'] = distance(
        keypoints[:, idx["left_wrist"]], 
        keypoints[:, idx["right_wrist"]]
    )
    
    # Distance between feet
    distances['foot_to_foot'] = distance(
        keypoints[:, idx["left_ankle"]], 
        keypoints[:, idx["right_ankle"]]
    )
    
    # Torso length (shoulder to hip)
    distances['left_torso_length'] = distance(
        keypoints[:, idx["left_shoulder"]], 
        keypoints[:, idx["left_hip"]]
    )
    
    distances['right_torso_length'] = distance(
        keypoints[:, idx["right_shoulder"]], 
        keypoints[:, idx["right_hip"]]
    )
    
    return distances

def normalize_keypoints(keypoints):
    """
    Normalize keypoints to be scale and position invariant.
    - Center each frame on the pelvis (hip center)
    - Scale by average torso length to make scale-invariant
    """
    idx = {
        "left_shoulder": 0, "right_shoulder": 1, "left_elbow": 2, "right_elbow": 3,
        "left_wrist": 4, "right_wrist": 5, "left_hip": 6, "right_hip": 7,
        "left_knee": 8, "right_knee": 9, "left_ankle": 10, "right_ankle": 11
    }
    
    normalized_keypoints = keypoints.copy()
    
    for frame_idx in range(keypoints.shape[0]):
        frame = keypoints[frame_idx]
        
        # Skip frames with insufficient data
        valid_mask = ~np.isnan(frame).any(axis=1)
        if valid_mask.sum() < 4:  # Need at least 4 valid keypoints
            continue
            
        # Calculate pelvis center (hip center)
        left_hip = frame[idx["left_hip"]]
        right_hip = frame[idx["right_hip"]]
        
        if not (np.isnan(left_hip).any() or np.isnan(right_hip).any()):
            pelvis_center = (left_hip + right_hip) / 2
            
            # Center all keypoints on pelvis
            centered_frame = frame - pelvis_center
            
            # Calculate torso lengths for scaling
            left_shoulder = frame[idx["left_shoulder"]]
            right_shoulder = frame[idx["right_shoulder"]]
            
            torso_lengths = []
            if not (np.isnan(left_shoulder).any() or np.isnan(left_hip).any()):
                torso_lengths.append(np.linalg.norm(left_shoulder - left_hip))
            if not (np.isnan(right_shoulder).any() or np.isnan(right_hip).any()):
                torso_lengths.append(np.linalg.norm(right_shoulder - right_hip))
            
            if torso_lengths:
                avg_torso_length = np.mean(torso_lengths)
                if avg_torso_length > 0:
                    # Scale by torso length to normalize size
                    normalized_frame = centered_frame / avg_torso_length
                    normalized_keypoints[frame_idx] = normalized_frame
    
    return normalized_keypoints

def numpy_to_list(arr):
    if isinstance(arr, np.ndarray):
        arr_clean = np.where(np.isnan(arr), None, arr)
        return arr_clean.tolist()
    return arr


def process_keypoints_sequence(keypoints: np.ndarray, frames_meta: list):
    """
    Process a (frames, joints, 2) array for one fencer.
    frames_meta: list of dicts with keys frame_number, interpolated.
    """
    # Convert zeros to NaN for better math (zeros indicate no detection)
    zero_mask = (keypoints == 0).all(axis=-1)
    keypoints[zero_mask] = np.nan

    # Drop frames that contain any NaN or inf
    finite_mask = np.all(np.isfinite(keypoints), axis=(1, 2))
    keypoints = keypoints[finite_mask]
    frames_meta = [m for m, keep in zip(frames_meta, finite_mask) if keep]

    if keypoints.size == 0 or len(frames_meta) == 0:
        return {
            "body_keypoint_names": BODY_KEYPOINT_NAMES,
            "num_frames": 0,
            "num_keypoints": len(BODY_KEYPOINT_NAMES),
            "frame_data": [],
        }

    normalized_keypoints = normalize_keypoints(keypoints)
    angles = compute_angles(keypoints)
    distances = compute_distances(keypoints)

    processed_data = {
        "body_keypoint_names": BODY_KEYPOINT_NAMES,
        "num_frames": len(frames_meta),
        "num_keypoints": len(BODY_KEYPOINT_NAMES),
        "frame_data": [],
    }
    for i, meta in enumerate(frames_meta):
        frame_data = {
            "frame_number": meta["frame_number"],
            "interpolated": meta["interpolated"],
            "keypoints": numpy_to_list(keypoints[i]),
            "normalized_keypoints": numpy_to_list(normalized_keypoints[i]),
            "angles": {
                "left_elbow": float(angles["left_elbow"][i]) if not np.isnan(angles["left_elbow"][i]) else None,
                "right_elbow": float(angles["right_elbow"][i]) if not np.isnan(angles["right_elbow"][i]) else None,
                "left_knee": float(angles["left_knee"][i]) if not np.isnan(angles["left_knee"][i]) else None,
                "right_knee": float(angles["right_knee"][i]) if not np.isnan(angles["right_knee"][i]) else None,
            },
            "distances": {
                "hand_to_hand": float(distances["hand_to_hand"][i]) if not np.isnan(distances["hand_to_hand"][i]) else None,
                "foot_to_foot": float(distances["foot_to_foot"][i]) if not np.isnan(distances["foot_to_foot"][i]) else None,
                "left_torso_length": float(distances["left_torso_length"][i]) if not np.isnan(distances["left_torso_length"][i]) else None,
                "right_torso_length": float(distances["right_torso_length"][i]) if not np.isnan(distances["right_torso_length"][i]) else None,
            },
        }
        processed_data["frame_data"].append(frame_data)
    return processed_data


def load_and_split(json_path: Path):
    """Load v3 json and split into per-fencer arrays."""
    with open(json_path, "r") as f:
        frames = json.load(f)
    left_list, right_list = [], []
    frames_meta = []
    for frame in frames:
        lkp, rkp = None, None
        kp_pair = frame.get("keypoints", [])
        if len(kp_pair) > 0 and kp_pair[0] is not None:
            lkp = np.array(kp_pair[0], dtype=np.float32)
        if len(kp_pair) > 1 and kp_pair[1] is not None:
            rkp = np.array(kp_pair[1], dtype=np.float32)
        left_list.append(lkp)
        right_list.append(rkp)
        frames_meta.append(
            {
                "frame_number": frame.get("frame", None),
                "interpolated": False,  # v3 frames do not flag per-fencer interpolation
            }
        )
    return left_list, right_list, frames_meta


def prepare_side(kp_list: list, flip_width: float = None):
    """Convert list of per-frame keypoints to array and strip head joints."""
    # Replace missing frames with NaNs
    arr = []
    for kp in kp_list:
        if kp is None:
            arr.append(np.full((17, 2), np.nan, dtype=np.float32))
        else:
            arr.append(kp)
    keypoints = np.array(arr, dtype=np.float32)  # (F, 17, 2)
    if flip_width is not None:
        keypoints[..., 0] = np.where(np.isnan(keypoints[..., 0]), np.nan, flip_width - keypoints[..., 0])
    # Remove head joints
    keypoints = keypoints[:, HEAD_JOINTS:, :]
    return keypoints


def get_frame_width(json_path: Path) -> float:
    """Try to read frame width from the rendered pose mp4 next to the JSON."""
    mp4_path = json_path.with_suffix(".mp4")
    if not mp4_path.exists():
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return w if w > 0 else None

def main():
    parser = argparse.ArgumentParser(description="Post-process extracted left/right pose JSONs.")
    parser.add_argument("--pose-overlay-dir", default=str(DEFAULT_POSE_OVERLAY_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    pose_overlay_dir = Path(args.pose_overlay_dir)
    output_root = Path(args.output_root)
    output_left = output_root / "left"
    output_right = output_root / "right"
    output_left_from_right = output_root / "left_from_right"
    for p in (output_left, output_right, output_left_from_right):
        p.mkdir(parents=True, exist_ok=True)

    """Process all JSON files in the pose_overlay directory (recursive)."""
    json_files = list(pose_overlay_dir.glob("**/*_pose.json"))
    if not json_files:
        logging.error(f"No *_pose.json files found in {pose_overlay_dir}")
        return
    logging.info(f"Found {len(json_files)} JSON files to process")
    processed = 0
    failed = 0
    for json_file in json_files:
        try:
            rel_dir = json_file.parent.relative_to(pose_overlay_dir)
            left_dir = output_left / rel_dir
            right_dir = output_right / rel_dir
            left_from_right_dir = output_left_from_right / rel_dir
            for d in (left_dir, right_dir, left_from_right_dir):
                d.mkdir(parents=True, exist_ok=True)

            logging.info(f"Processing {json_file}")
            left_list, right_list, frames_meta = load_and_split(json_file)
            frame_width = get_frame_width(json_file)

            left_kp = prepare_side(left_list)
            right_kp = prepare_side(right_list)
            right_flipped_kp = prepare_side(right_list, flip_width=frame_width) if frame_width else None

            if left_kp.size:
                data_left = process_keypoints_sequence(left_kp, frames_meta)
                out_left = left_dir / f"{json_file.stem}_processed.json"
                with open(out_left, "w") as f:
                    json.dump(data_left, f, indent=2)
                logging.info(f"✔ Saved {out_left}")

            if right_kp.size:
                data_right = process_keypoints_sequence(right_kp, frames_meta)
                out_right = right_dir / f"{json_file.stem}_processed.json"
                with open(out_right, "w") as f:
                    json.dump(data_right, f, indent=2)
                logging.info(f"✔ Saved {out_right}")

            if right_flipped_kp is not None and right_flipped_kp.size:
                data_left_from_right = process_keypoints_sequence(right_flipped_kp, frames_meta)
                out_lfr = left_from_right_dir / f"{json_file.stem}_processed.json"
                with open(out_lfr, "w") as f:
                    json.dump(data_left_from_right, f, indent=2)
                logging.info(f"✔ Saved {out_lfr}")

            processed += 1
        except Exception as e:
            logging.error(f"✗ Error processing {json_file}: {e}")
            failed += 1
    logging.info(f"Processing complete: {processed} successful, {failed} failed")
    logging.info(f"Processed data saved to: {output_root}")

if __name__ == "__main__":
    main()
