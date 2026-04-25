"""Unit tests for HHA encoding (`src/data_utils/hha/`).

Tests verify the canonical SUN RGB-D toolbox conventions:
    1. Synthetic ground plane: height ~ 0, angle ~ 0 (normal aligned with
       anti-gravity), disparity ~ 1/Z.
    2. Synthetic vertical wall: angle ~ 90 (normal perpendicular to gravity).
    3. Synthetic 45-degree ramp: angle ~ 45.
    4. Hflip equivariance for all three channels.
    5. Failure mode: singular K returns all-NaN tensor without raising.
    6. Real-data smoke: load one SUN RGB-D sample, assert plausible ranges.
"""

import os

import numpy as np
import pytest

from src.data_utils.hha import compute_hha, read_extrinsics, read_intrinsics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_K(fx: float = 500.0, fy: float = 500.0,
            cx: float = 64.0, cy: float = 48.0) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _identity_R_tilt() -> np.ndarray:
    """An R_tilt that maps intermediate-camera (X_cam, Z_cam, -Y_cam) to
    world (X_world, Y_world, Z_world) with Y up.

    With identity, the toolbox-intermediate (X, Z, -Y) frame IS the world
    frame: world X = X_cam, world Y = Z_cam (depth), world Z = -Y_cam.

    For the synthetic tests below this means:
        - "world Y" is the camera's depth axis (forward).
        - A frontal plane (constant depth Z) has constant world-Y.
        - A floor (depth varying with image row) has world-Z varying with
          camera Y_cam, so it's not a "ground" plane in this frame.

    To build clean synthetic tests we therefore craft scenes where the
    "floor" is whatever surface has its normal pointing along world +Y.
    With identity R_tilt, world +Y = camera +Z (forward), so a frontal
    plane (constant depth) has its normal along world +Y, i.e. it acts as
    a "horizontal surface facing up" in world coordinates.
    """
    return np.eye(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# Synthetic geometry tests
# ---------------------------------------------------------------------------

def test_ground_plane_constant_depth():
    """Frontal plane at depth Z: with identity R_tilt this acts as a 'floor'
    (normal along world +Y). Disparity = 1/Z, height ~ 0, angle ~ 0.
    """
    H, W = 64, 96
    Z = 2.0  # meters
    depth = np.full((H, W), Z, dtype=np.float32)
    K = _make_K(cx=W / 2.0, cy=H / 2.0)
    R = _identity_R_tilt()

    hha = compute_hha(depth, K, R)
    assert hha.shape == (3, H, W)
    assert hha.dtype == np.float32

    # Inspect the inner region to avoid border effects from edge-padded normals.
    inner = (slice(None), slice(8, H - 8), slice(8, W - 8))
    disparity, height, angle = hha[inner[0]][:, inner[1], inner[2]], None, None  # noqa
    disp = hha[0, 8:H - 8, 8:W - 8]
    h = hha[1, 8:H - 8, 8:W - 8]
    ang = hha[2, 8:H - 8, 8:W - 8]

    np.testing.assert_allclose(disp, 1.0 / Z, rtol=1e-5)
    np.testing.assert_allclose(h, 0.0, atol=0.1)  # ~0 after floor subtract
    # Angle should be ~0 deg (normal aligned with anti-gravity).
    assert np.nanmean(ang) < 5.0, f"expected angle ~0, got mean {np.nanmean(ang):.2f}"


def test_vertical_wall_depth_varies_with_column():
    """Wall at world X = const (a side wall): normal lies along world X,
    perpendicular to anti-gravity (world +Y), so angle ~ 90.

    The synthetic image is wide and uses a small focal length so the
    unsaturated depth range covers a meaningful column band; pixels outside
    that band are marked invalid (depth = 0) and excluded from the test.
    """
    H, W = 80, 320
    fx = fy = 60.0
    cx, cy = W / 2.0, H / 2.0
    K = _make_K(fx, fy, cx, cy)
    R = _identity_R_tilt()

    X_world = 1.5  # 1.5 m to the right
    # depth = X_world * fx / (u - cx). Restrict u such that depth in [1, 6] m.
    u = np.arange(W, dtype=np.float64)[None, :]
    du = u - cx
    valid_cols = (du > X_world * fx / 6.0) & (du < X_world * fx / 1.0)
    safe_du = np.where(valid_cols, du, 1.0)
    d_vals = X_world * fx / safe_du
    depth = np.where(valid_cols, d_vals, 0.0).astype(np.float32)
    depth = np.broadcast_to(depth, (H, W)).copy()

    hha = compute_hha(depth, K, R)
    ang = hha[2]
    # Test only well inside the wall region (avoid borders where the
    # finite-difference normal estimator is unreliable).
    valid_2d = np.broadcast_to(valid_cols, (H, W))
    test_mask = np.zeros_like(valid_2d, dtype=bool)
    test_mask[8:H - 8, :] = valid_2d[8:H - 8, :]
    # Erode the column mask by a few pixels to avoid the wall's left/right
    # edges (where one neighbor is invalid).
    col_mask = valid_cols[0]
    cols = np.where(col_mask)[0]
    if cols.size >= 12:
        cols = cols[6:-6]
        col_keep = np.zeros_like(col_mask)
        col_keep[cols] = True
        test_mask &= col_keep[None, :]

    finite_ang = ang[test_mask & np.isfinite(ang)]
    assert finite_ang.size > 50, f"too few unsaturated wall pixels ({finite_ang.size})"
    median_ang = float(np.median(finite_ang))
    assert median_ang > 75.0, (
        f"expected vertical-wall angle near 90, got median {median_ang:.2f}"
    )


def test_45_degree_ramp():
    """A surface tilted 45 degrees about the camera-X axis. Its normal in
    the world frame makes a 45 degree angle with anti-gravity.

    With identity R_tilt and intermediate frame (X_cam, Z_cam, -Y_cam):
        world Y = Z_cam, world Z = -Y_cam.
    A surface with normal n_world = (0, cos(45), sin(45)) corresponds to
    a plane in the intermediate frame whose Z and -Y_cam components are
    related by tan(45) = 1, i.e., a plane where Z_cam = -Y_cam + const.
    Pixels along a row (constant Y_cam) have constant Z_cam — but rows
    differ.  So depth(u, v) should depend only on v: a ramp where deeper
    rows are nearer.
    """
    H, W = 64, 96
    fx = fy = 500.0
    cx, cy = W / 2.0, H / 2.0
    K = _make_K(fx, fy, cx, cy)
    R = _identity_R_tilt()

    # Plane in intermediate frame: X_cam, Z_cam = -Y_cam + c → Z_cam + Y_cam = c.
    # With Y_cam = (v - cy) * Z / fy and we want (Z + (v - cy) * Z / fy) = c
    # → Z * (1 + (v - cy)/fy) = c → Z(v) = c / (1 + (v - cy)/fy).
    v = np.arange(H, dtype=np.float64)[:, None]
    c = 4.0  # meters
    Z_row = c / (1.0 + (v - cy) / fy)
    Z_row = np.clip(Z_row, 0.5, 7.5)
    depth = np.broadcast_to(Z_row, (H, W)).astype(np.float32)

    hha = compute_hha(depth, K, R)
    ang = hha[2, 16:H - 16, 16:W - 16]
    finite_ang = ang[np.isfinite(ang)]
    assert finite_ang.size > 0
    median_ang = float(np.median(finite_ang))
    # Allow a generous tolerance because finite-difference normals are noisy.
    assert 30.0 < median_ang < 60.0, (
        f"expected ramp angle ~45, got median {median_ang:.2f}"
    )


# ---------------------------------------------------------------------------
# Hflip equivariance (Phase 1 test #4)
# ---------------------------------------------------------------------------

def test_hflip_equivariance_all_channels():
    """compute_hha(hflip(depth)) ~= hflip(compute_hha(depth)) for every channel.

    Disparity and height are hflip-equivariant by construction (no horizontal
    dependence). The angle channel is hflip-equivariant specifically because
    we use ``arccos(|n_y|)``: the absolute value collapses left/right symmetric
    normals to the same angle.

    We use a smooth synthetic scene (the 45-degree ramp) to avoid numerical
    instability at depth discontinuities.
    """
    H, W = 64, 96
    fx = fy = 500.0
    cx, cy = W / 2.0, H / 2.0
    K = _make_K(fx, fy, cx, cy)
    R = _identity_R_tilt()

    v = np.arange(H, dtype=np.float64)[:, None]
    Z_row = 4.0 / (1.0 + (v - cy) / fy)
    Z_row = np.clip(Z_row, 0.5, 7.5)
    depth = np.broadcast_to(Z_row, (H, W)).astype(np.float32)

    # Add a horizontally-varying perturbation so the test is non-trivial.
    u = np.arange(W, dtype=np.float64)[None, :]
    perturbation = 0.05 * np.sin(2.0 * np.pi * u / W)
    depth = (depth + perturbation).astype(np.float32)

    hha = compute_hha(depth, K, R)
    hha_of_flipped = compute_hha(np.ascontiguousarray(depth[:, ::-1]), K, R)
    flipped_hha = hha[:, :, ::-1]

    # Compare on the inner region (avoid border effects).
    inner = (slice(None), slice(8, H - 8), slice(8, W - 8))
    a = hha_of_flipped[inner]
    b = flipped_hha[inner]
    # NaN locations should match.
    assert np.array_equal(np.isnan(a), np.isnan(b))
    finite = ~np.isnan(a)
    np.testing.assert_allclose(a[finite], b[finite], atol=2.0, rtol=1e-3,
                               err_msg="HHA must be hflip-equivariant on all channels")


# ---------------------------------------------------------------------------
# Failure mode (Phase 1 test #6)
# ---------------------------------------------------------------------------

def test_singular_K_returns_all_nan(caplog):
    H, W = 32, 48
    depth = np.full((H, W), 2.0, dtype=np.float32)
    K_singular = np.zeros((3, 3), dtype=np.float64)  # fx = fy = 0 → divide by zero
    R = _identity_R_tilt()

    # The function should *not* raise — it should log a warning and return
    # an all-invalid tensor. (Division by zero may give inf/nan rather than
    # raising, so we accept either an exception path or a structured-NaN
    # path; both must produce no surviving finite pixels in the angle
    # channel.)
    with caplog.at_level("WARNING"):
        hha = compute_hha(depth, K_singular, R)
    assert hha.shape == (3, H, W)
    assert hha.dtype == np.float32
    # Every angle pixel should be NaN (no valid normals possible).
    assert np.all(np.isnan(hha[2]))


def test_invalid_value_substitution():
    """Custom invalid_value (instead of NaN) is honored at missing pixels."""
    H, W = 32, 48
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[:8, :8] = 0.0  # mark a hole
    K = _make_K(cx=W / 2.0, cy=H / 2.0)
    R = _identity_R_tilt()

    hha = compute_hha(depth, K, R, invalid_value=-1.0)
    # The hole should carry -1.0 in all three channels (no NaN).
    assert not np.isnan(hha[:, :8, :8]).any()
    np.testing.assert_array_equal(hha[:, :8, :8], -1.0)


# ---------------------------------------------------------------------------
# Real-data smoke (Phase 1 test #5)
# ---------------------------------------------------------------------------

REAL_SCENE = (
    "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/"
    "data/sunrgbd/SUNRGBD/kv1/NYUdata/NYU0191"
)


@pytest.mark.skipif(not os.path.isdir(REAL_SCENE), reason="real SUN RGB-D scene not present")
def test_real_sample_smoke():
    """Load one real SUN RGB-D sample and check plausible HHA ranges."""
    from PIL import Image

    K = read_intrinsics(REAL_SCENE)
    R = read_extrinsics(REAL_SCENE)

    depth_files = sorted(os.listdir(os.path.join(REAL_SCENE, "depth_bfx")))
    assert depth_files, "no depth_bfx PNGs found"
    depth_path = os.path.join(REAL_SCENE, "depth_bfx", depth_files[0])
    raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    # SUN RGB-D 3-bit shift unpacking, then mm -> meters.
    unpacked = np.bitwise_or(raw >> 3, raw << 13).astype(np.uint16)
    depth_m = unpacked.astype(np.float32) / 1000.0

    hha = compute_hha(depth_m, K, R)
    assert hha.shape == (3, depth_m.shape[0], depth_m.shape[1])

    finite = ~np.isnan(hha)
    nan_rate_per_channel = 1.0 - finite.mean(axis=(1, 2))
    # NaN rate per channel should be modest (most of the image is valid).
    assert nan_rate_per_channel.max() < 0.5, (
        f"unexpectedly high NaN rate: {nan_rate_per_channel}"
    )

    # Plausible value ranges per channel.
    disp = hha[0][np.isfinite(hha[0])]
    height = hha[1][np.isfinite(hha[1])]
    angle = hha[2][np.isfinite(hha[2])]
    assert disp.size > 0 and height.size > 0 and angle.size > 0
    assert 0.0 <= disp.min() and disp.max() <= 10.0 + 1e-6
    assert -3.0 <= np.percentile(height, 1) and np.percentile(height, 99) <= 5.0
    assert 0.0 <= angle.min() and angle.max() <= 90.0 + 1e-3


# ---------------------------------------------------------------------------
# Intrinsics / extrinsics IO sanity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(REAL_SCENE), reason="real SUN RGB-D scene not present")
def test_intrinsics_read_shape():
    K = read_intrinsics(REAL_SCENE)
    assert K.shape == (3, 3)
    assert K[0, 0] > 0 and K[1, 1] > 0
    assert K[2, 2] == 1.0


@pytest.mark.skipif(not os.path.isdir(REAL_SCENE), reason="real SUN RGB-D scene not present")
def test_extrinsics_read_shape_and_orthogonality():
    R = read_extrinsics(REAL_SCENE)
    assert R.shape == (3, 3)
    # Should be approximately orthogonal (R @ R.T == I).
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=5e-2)
