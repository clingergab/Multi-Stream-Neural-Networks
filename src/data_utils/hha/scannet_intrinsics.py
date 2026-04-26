"""ScanNet-specific intrinsics + R_tilt extraction for HHA preprocessing.

ScanNet has a different on-disk layout than SUN RGB-D:

  <scene>/intrinsic/intrinsic_depth.txt    -- 4x4 K (top-left 3x3 used)
  <scene>/pose/<frame_idx>.txt              -- 4x4 camera_to_world per frame
  <scene>/<scene>.txt                       -- contains axisAlignment (4x4)

The math composition for HHA is:

    R_scannet = orthogonalize(axisAlignment[:3, :3]) @ pose[:3, :3]

which maps a camera-frame point directly into the gravity-aligned world frame
(+Z up). Pass this matrix to ``compute_hha`` with
``apply_sunrgbd_basis_swap=False``.

axisAlignment is in general NOT a pure rotation -- it can have per-axis scale
folded in. ``_orthogonalize`` SVD-decomposes it and returns the nearest pure
rotation (det = +1).

Whether axisAlignment is stored as 'aligned <- raw' or 'raw <- aligned' is
decided ONCE per dataset version in Phase 0 (see notebooks/scannet_phase0_validation.ipynb)
and passed explicitly through the API as ``inverted=...``. No module-level
state.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


def _orthogonalize(M: np.ndarray) -> np.ndarray:
    """SVD-orthogonalize a 3x3 matrix that may have scale folded in.

    Returns the nearest pure ROTATION (det = +1, no reflection).

    SVD's ``U @ Vt`` can return det = -1 when M includes a reflection.
    If that happens we flip the last row of Vt before recomposing -- otherwise
    the height channel comes out sign-flipped.
    """
    if M.shape != (3, 3):
        raise ValueError(f"_orthogonalize expects a 3x3 matrix, got {M.shape}")
    U, _S, Vt = np.linalg.svd(M.astype(np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = U @ Vt
    return R


def read_intrinsic_depth(scene_dir: str) -> np.ndarray:
    """Load the 3x3 depth-intrinsics matrix K for a ScanNet scene.

    ScanNet stores it as the top-left 3x3 block of a 4x4 in
    ``<scene_dir>/intrinsic/intrinsic_depth.txt`` (16 whitespace-separated floats,
    row-major).
    """
    path = os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
    with open(path, "r") as f:
        vals = np.fromstring(f.read(), sep=" ", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(
            f"Expected 16 floats in {path}, got {vals.size}"
        )
    K_full = vals.reshape(4, 4)
    return K_full[:3, :3].copy()


def read_axis_alignment(
    scene_dir: str,
    *,
    inverted: bool = False,
) -> Optional[np.ndarray]:
    """Read the axisAlignment matrix for a ScanNet scene.

    ScanNet stores it as a single ``axisAlignment = ...`` line in
    ``<scene_dir>/<scene_id>.txt`` (the basename of scene_dir is the scene id).
    Returns the 4x4 matrix if present, ``None`` if missing (caller logs a
    warning and falls back to identity for that scene).

    Args:
        scene_dir: Path to the ScanNet scene directory.
        inverted: If True, the dataset stores axisAlignment in the
            ``raw <- aligned`` direction rather than the canonical
            ``aligned <- raw``. Pass ``True`` only if Phase 0's relative-floor
            test on the 200-scene scan determined the convention is inverted
            for this ScanNet release. The function will pre-transpose the
            matrix's rotation block so the rest of the pipeline always sees
            the canonical 'aligned <- raw' direction.

    No module-level state -- the caller threads the flag through explicitly.
    """
    scene_id = os.path.basename(os.path.normpath(scene_dir))
    path = os.path.join(scene_dir, f"{scene_id}.txt")
    if not os.path.isfile(path):
        return None
    matrix: Optional[np.ndarray] = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("axisAlignment"):
                # ``axisAlignment = a b c d e f g h i j k l m n o p``
                m = re.match(r"axisAlignment\s*=\s*(.+)$", line)
                if m is None:
                    continue
                vals = np.fromstring(m.group(1), sep=" ", dtype=np.float64)
                if vals.size == 16:
                    matrix = vals.reshape(4, 4)
                break
    if matrix is None:
        return None
    if inverted:
        # Canonicalize to 'aligned <- raw': transpose just the rotation block.
        rot = matrix[:3, :3].T
        out = matrix.copy()
        out[:3, :3] = rot
        return out
    return matrix


def read_pose(scene_dir: str, frame_idx: int) -> Optional[np.ndarray]:
    """Read the 4x4 camera-to-world pose for a frame.

    ``<scene_dir>/pose/<frame_idx>.txt`` contains 16 whitespace-separated floats
    (row-major 4x4). ScanNet's BundleFusion sometimes emits ``-inf`` / NaN for
    frames where tracking failed; in that case we return ``None`` and the
    caller skips the frame (counted toward the failure-rate gate).
    """
    path = os.path.join(scene_dir, "pose", f"{frame_idx}.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        vals = np.fromstring(f.read(), sep=" ", dtype=np.float64)
    if vals.size != 16:
        return None
    pose = vals.reshape(4, 4)
    if not np.all(np.isfinite(pose)):
        return None
    return pose


@dataclass
class ScannetSceneMeta:
    """Per-scene metadata, computed once during preprocessing and reused per frame."""
    scene_id: str
    K: np.ndarray                      # 3x3 depth intrinsics for native resolution
    axis_alignment_rot: np.ndarray     # 3x3 orthogonalized rotation, det = +1
    depth_shift: float                 # from the .sens header (almost always 1000.0)
    reference_R_scannet: np.ndarray    # 3x3 reference rotation (median frame) for drift comparison
    orthogonality_residual: float      # ||axisAlignment[:3,:3] - axis_alignment_rot||_F


def compute_scannet_rtilt(
    meta: ScannetSceneMeta,
    pose: np.ndarray,
) -> Optional[np.ndarray]:
    """Returns ``R_scannet = meta.axis_alignment_rot @ pose[:3, :3]`` suitable for
    ``compute_hha(..., apply_sunrgbd_basis_swap=False)``.

    ``pose`` must be a 4x4 camera-to-scene-local-world transform; only the
    rotation block ``pose[:3, :3]`` is used (translation drops out for a
    rotation-only R_tilt).

    Returns ``None`` if ``pose`` is invalid (shape wrong or contains non-finite
    values); the caller should skip the frame and increment the failure count.

    ``meta.axis_alignment_rot`` was already canonicalized to 'aligned <- raw'
    by ``read_axis_alignment(..., inverted=...)`` at scene-cache time, so this
    function takes no ``inverted`` flag -- the canonicalization happens once
    upstream and the explicit threading stops there.
    """
    if pose is None or pose.shape != (4, 4):
        return None
    pose_rot = pose[:3, :3]
    if not np.all(np.isfinite(pose_rot)):
        return None
    return meta.axis_alignment_rot @ pose_rot


def angular_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance between two SO(3) rotations in degrees.

    Use this for the per-scene drift gate: pick the median frame's R_scannet
    as the scene reference, then measure each other frame's deviation. Median
    frame is robust enough for the > 15 degree flag and avoids iterating a
    Karcher mean (Frobenius mean of rotations is NOT a rotation in general
    when there's a few-degree spread).
    """
    M = R_a @ R_b.T
    cos_theta = (np.trace(M) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def load_drop_list(drop_list_path: str) -> dict:
    """Load Phase 0's drop-list JSON.

    Expected schema:

        {
            "convention_verified": true,
            "axis_alignment_inverted": false,
            "scenes": {
                "scene0023_01": "broken_axisAlignment_relative_floor=0.74",
                "scene0079_02": "missing_axisAlignment",
                ...
            }
        }

    Phase 3's preprocess notebook reads this once at the top.
    """
    with open(drop_list_path, "r") as f:
        return json.load(f)


def should_drop(scene_id: str, drop_list: dict) -> bool:
    """True if a scene should be excluded from preprocessing.

    Scenes flagged with ``missing_axisAlignment`` are NOT dropped -- they
    fall back to identity at preprocessing time (the missing-file fallback
    in ``read_axis_alignment`` returning ``None``). Only ``broken_*``,
    ``high_nan_rate=*``, and ``pose_drift_*`` scenes are dropped.
    """
    reason = drop_list.get("scenes", {}).get(scene_id)
    if not reason:
        return False
    return not reason.startswith("missing_axisAlignment")
