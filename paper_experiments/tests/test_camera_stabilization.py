from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stabilize_processed_pose import cumulative_transforms, stabilize_payload


def camera_with_translation(dx: float, frames: int = 4) -> dict:
    motions = [
        {
            "frame": 0,
            "valid": False,
            "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    ]
    motions.extend(
        {
            "frame": index,
            "valid": True,
            "affine": [[1.0, 0.0, dx], [0.0, 1.0, 0.0]],
        }
        for index in range(1, frames)
    )
    return {
        "frame_count": frames,
        "source_width": 100,
        "source_height": 50,
        "analysis_width": 100,
        "analysis_height": 50,
        "motions": motions,
    }


def test_translation_stabilization_removes_camera_pan() -> None:
    cumulative = cumulative_transforms(camera_with_translation(3.0), "translation")
    payload = {
        "body_keypoint_names": ["joint"],
        "frame_data": [
            {"frame_number": frame, "keypoints": [[10.0 + 3.0 * frame, 20.0]]}
            for frame in range(4)
        ],
    }

    stabilized = stabilize_payload(payload, cumulative, "translation")
    x = [frame["keypoints"][0][0] for frame in stabilized["frame_data"]]

    np.testing.assert_allclose(x, [10.0, 10.0, 10.0, 10.0], atol=1e-6)


def test_stabilization_preserves_motion_relative_to_reference_frame() -> None:
    cumulative = cumulative_transforms(camera_with_translation(2.0, frames=5), "translation")
    payload = {
        "body_keypoint_names": ["joint"],
        "frame_data": [
            # Pose processing can begin after video frame zero.
            {"frame_number": frame, "keypoints": [[5.0 + 2.0 * frame + (frame - 2), 8.0]]}
            for frame in range(2, 5)
        ],
    }

    stabilized = stabilize_payload(payload, cumulative, "translation")
    x = [frame["keypoints"][0][0] for frame in stabilized["frame_data"]]

    np.testing.assert_allclose(x, [9.0, 10.0, 11.0], atol=1e-6)
