"""Synthetic-only tests for the ScanNet HHA path.

ScanNet's raw `.sens` data is Drive-only -- never on local disk -- so these
tests cover the math, intrinsics math, and helper APIs only. The real-data
shape + range contract is enforced by the preprocess notebook's validation
gate (Phase 3.6 in the plan), not pytest.
"""

import os
import tempfile

import numpy as np
import pytest

from src.data_utils.hha import (
    ScannetSceneMeta,
    angular_distance_deg,
    compute_hha,
    compute_scannet_rtilt,
    load_drop_list,
    read_axis_alignment,
    read_intrinsic_depth,
    read_pose,
    should_drop,
)
from src.data_utils.hha.scannet_intrinsics import _orthogonalize


# ---------------------------------------------------------------------------
# _orthogonalize: rotation x scale and rotation x reflection x scale recovery
# ---------------------------------------------------------------------------

def test_orthogonalize_recovers_rotation_after_scale():
    """A rotation with diagonal scale folded in should orthogonalize back to
    the original rotation within float tolerance."""
    rng = np.random.default_rng(0)
    # Random rotation via QR.
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    # Add per-axis scale.
    S = np.diag([0.97, 1.05, 1.02])
    M = Q @ S
    R = _orthogonalize(M)
    assert np.linalg.det(R) > 0
    np.testing.assert_allclose(R, Q, atol=1e-6)


def test_orthogonalize_corrects_reflection():
    """A rotation x reflection x scale matrix should still come back as a
    pure rotation (det = +1) after orthogonalization."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    reflection = np.diag([1.0, 1.0, -1.0])
    S = np.diag([1.01, 0.99, 1.0])
    M = Q @ reflection @ S
    R = _orthogonalize(M)
    # Result must be a proper rotation, regardless of input chirality.
    assert np.linalg.det(R) > 0


# ---------------------------------------------------------------------------
# Synthetic compute_hha with apply_sunrgbd_basis_swap=False
# ---------------------------------------------------------------------------

def _make_K(fx=200.0, fy=200.0, cx=128.0, cy=128.0):
    return np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
        dtype=np.float64,
    )


def test_scannet_floor_angle_180():
    """A horizontal floor (world-Z = const < 0) under identity R_tilt with
    apply_sunrgbd_basis_swap=False should yield angle ~ 180 deg (canonical
    signed HHA: floor normal points up, opposite to gravity)."""
    H, W = 80, 96
    fx = fy = 200.0
    cx, cy = W / 2.0, H / 2.0
    K = _make_K(fx, fy, cx, cy)
    R = np.eye(3, dtype=np.float64)

    # Identity R_scannet means world == camera frame. Camera's +Y is "down"
    # in OpenCV convention. To put a floor BELOW the camera (world-Z = -1)
    # while using world == camera, we need the camera's "down" axis to be
    # equivalent to world-Z. With apply_sunrgbd_basis_swap=False the world
    # axes ARE the camera axes, so a floor at "below camera" = camera-Y > 0.
    # We synthesize the depth such that camera-Y = +1m (i.e. y_cam = +1).
    floor_y_cam = 1.0
    v = np.arange(H, dtype=np.float64)[:, None]
    valid_rows = v > cy + 1
    d_v = np.where(valid_rows, fy * floor_y_cam / np.maximum(v - cy, 1e-3), 0.0)
    depth = np.broadcast_to(d_v, (H, W)).astype(np.float32)
    depth = np.where(depth > 7.5, 0.0, depth).astype(np.float32)

    hha = compute_hha(depth, K, R, apply_sunrgbd_basis_swap=False)
    # In camera frame world (since R=I and no basis swap): "up" = camera +Z
    # (forward) -- NOT +Y. The actual canonical floor case for ScanNet uses
    # axisAlignment that rotates camera-down (Y_cam) to world +Z.
    # So this construction tests the formula application, not the geometric
    # alignment expectation. For the canonical ceiling/floor/wall regimes
    # we use an R_tilt that rotates camera-Y down to world-Z up:
    #   R_tilt = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    # which is exactly _M_LEFT.

    # Re-run with the correct R_tilt that aligns camera (X right, Y down,
    # Z forward) to world (X right, Y forward, Z up).
    R_align = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    hha = compute_hha(depth, K, R_align, apply_sunrgbd_basis_swap=False)

    valid = depth > 0
    inner = np.zeros_like(valid)
    inner[H // 2 + 6:H - 6, 6:W - 6] = True
    region = inner & valid
    assert region.sum() > 100

    angle = hha[2][region]
    height = hha[1][region]
    finite = np.isfinite(angle) & np.isfinite(height)
    assert finite.sum() > 50

    # Floor: angle ~ 180, height ~ 0 (after floor-percentile subtract).
    median_angle = float(np.median(angle[finite]))
    median_height = float(np.median(height[finite]))
    assert median_angle > 170, (
        f"expected floor angle ~180 (apply_sunrgbd_basis_swap=False), got {median_angle:.2f}"
    )
    assert abs(median_height) < 0.05, (
        f"expected floor height ~0, got {median_height:.3f}"
    )


def test_scannet_frontal_wall_angle_90():
    """Constant-depth frontal plane (a wall facing the camera). With the
    same camera->world R_tilt as the floor test, a frontal wall has its
    normal pointing along world-Y (forward), which is perpendicular to
    world-Z (up) -> angle ~ 90 deg."""
    H, W = 64, 96
    Z = 2.0
    K = _make_K(cx=W / 2.0, cy=H / 2.0)
    R_align = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    depth = np.full((H, W), Z, dtype=np.float32)
    hha = compute_hha(depth, K, R_align, apply_sunrgbd_basis_swap=False)
    ang = hha[2, 8:H - 8, 8:W - 8]
    finite = ang[np.isfinite(ang)]
    assert finite.size > 0
    assert np.nanmean(finite) > 80.0, (
        f"expected frontal-wall angle ~90, got mean {np.nanmean(finite):.2f}"
    )


def test_scannet_backward_compat_sun_path_unchanged():
    """The default apply_sunrgbd_basis_swap=True should still produce
    SUN-path results (regression coverage for the refactor)."""
    H, W = 64, 96
    Z = 2.0
    K = _make_K(cx=W / 2.0, cy=H / 2.0)
    R = np.eye(3, dtype=np.float64)
    depth = np.full((H, W), Z, dtype=np.float32)
    hha = compute_hha(depth, K, R)  # default apply_sunrgbd_basis_swap=True
    # Frontal wall under SUN's basis swap with identity R: angle ~ 90.
    ang = hha[2, 8:H - 8, 8:W - 8]
    finite = ang[np.isfinite(ang)]
    assert finite.size > 0
    assert np.nanmean(finite) > 80.0


# ---------------------------------------------------------------------------
# angular_distance_deg
# ---------------------------------------------------------------------------

def test_angular_distance_identity_zero():
    R = np.eye(3, dtype=np.float64)
    assert angular_distance_deg(R, R) == pytest.approx(0.0, abs=1e-10)


def test_angular_distance_30_deg():
    """Rotate by 30 deg around the Z axis; angular distance should be 30."""
    theta = np.deg2rad(30.0)
    Rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]],
        dtype=np.float64,
    )
    R0 = np.eye(3)
    assert angular_distance_deg(Rz, R0) == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------------------
# IO helpers (use tempdir-based fake scenes)
# ---------------------------------------------------------------------------

def test_read_intrinsic_depth_round_trip():
    K = np.array(
        [[200.0, 0.0, 128.0, 0.0],
         [0.0, 200.0, 96.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        scene_dir = os.path.join(tmp, "scene0000_00")
        os.makedirs(os.path.join(scene_dir, "intrinsic"))
        with open(os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt"), "w") as f:
            f.write(" ".join(f"{v:.6f}" for v in K.flatten()))
        K3 = read_intrinsic_depth(scene_dir)
        np.testing.assert_allclose(K3, K[:3, :3])


def test_read_axis_alignment_present_and_inverted():
    raw = np.eye(4, dtype=np.float64)
    raw[:3, :3] = np.array(
        [[0.5, -0.866, 0.0],
         [0.866, 0.5, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        scene_id = "scene0001_00"
        scene_dir = os.path.join(tmp, scene_id)
        os.makedirs(scene_dir)
        with open(os.path.join(scene_dir, f"{scene_id}.txt"), "w") as f:
            f.write("colorWidth = 1296\n")
            f.write("axisAlignment = " + " ".join(f"{v:.6f}" for v in raw.flatten()) + "\n")
            f.write("colorHeight = 968\n")

        out_canonical = read_axis_alignment(scene_dir, inverted=False)
        out_inverted = read_axis_alignment(scene_dir, inverted=True)
    assert out_canonical is not None and out_inverted is not None
    np.testing.assert_allclose(out_canonical, raw)
    # When inverted: rotation block should be the transpose of the canonical.
    np.testing.assert_allclose(out_inverted[:3, :3], raw[:3, :3].T)


def test_read_axis_alignment_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        scene_id = "scene0099_99"
        scene_dir = os.path.join(tmp, scene_id)
        os.makedirs(scene_dir)
        # No <scene>.txt file at all.
        assert read_axis_alignment(scene_dir) is None


def test_read_pose_returns_none_on_nan():
    with tempfile.TemporaryDirectory() as tmp:
        scene_dir = os.path.join(tmp, "scene_x")
        os.makedirs(os.path.join(scene_dir, "pose"))
        # Write a 4x4 with one NaN
        vals = np.eye(4).flatten().astype(np.float64)
        vals[5] = float("nan")
        with open(os.path.join(scene_dir, "pose", "0.txt"), "w") as f:
            f.write(" ".join(f"{v}" for v in vals))
        assert read_pose(scene_dir, 0) is None


# ---------------------------------------------------------------------------
# Drop-list helpers
# ---------------------------------------------------------------------------

def test_drop_list_should_drop_logic():
    drop = {
        "convention_verified": True,
        "axis_alignment_inverted": False,
        "scenes": {
            "scene0001_00": "broken_axisAlignment_relative_floor=0.74",
            "scene0002_00": "missing_axisAlignment",
            "scene0003_00": "high_nan_rate=0.41",
            "scene0004_00": "pose_drift_42pct_frames>15deg",
        },
    }
    assert should_drop("scene0001_00", drop)  # broken
    assert not should_drop("scene0002_00", drop)  # missing -> fall back, don't drop
    assert should_drop("scene0003_00", drop)
    assert should_drop("scene0004_00", drop)
    assert not should_drop("scene_not_listed", drop)


# ---------------------------------------------------------------------------
# compute_scannet_rtilt edge cases
# ---------------------------------------------------------------------------

def test_compute_scannet_rtilt_returns_none_on_invalid_pose():
    meta = ScannetSceneMeta(
        scene_id="scene_x",
        K=np.eye(3),
        axis_alignment_rot=np.eye(3),
        depth_shift=1000.0,
        reference_R_scannet=np.eye(3),
        orthogonality_residual=0.0,
    )
    pose = np.eye(4)
    pose[0, 0] = float("nan")
    assert compute_scannet_rtilt(meta, pose) is None


def test_compute_scannet_rtilt_identity():
    """Identity pose + identity axisAlignment -> identity R_scannet."""
    meta = ScannetSceneMeta(
        scene_id="scene_x",
        K=np.eye(3),
        axis_alignment_rot=np.eye(3),
        depth_shift=1000.0,
        reference_R_scannet=np.eye(3),
        orthogonality_residual=0.0,
    )
    pose = np.eye(4)
    R = compute_scannet_rtilt(meta, pose)
    assert R is not None
    np.testing.assert_allclose(R, np.eye(3))
