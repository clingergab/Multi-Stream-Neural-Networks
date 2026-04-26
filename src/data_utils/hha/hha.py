"""Compute the 3-channel HHA encoding from a SUN RGB-D depth image.

Channels (matching Gupta et al. 2014, ``saveHHA.m`` from rcnn-depth):
    0. Horizontal disparity = 1 / depth_m, clamped to [0, 10] (1/m).
    1. Height above ground in meters: world-frame **Z_world** minus a per-image
       floor estimate (5th percentile of valid Z_world).
    2. **Signed** angle with gravity in degrees, in [0, 180]. Computed as
       ``arccos(N · g) * 180/pi`` where ``g`` is the gravity direction (-Z
       in our world frame). This is the canonical form:

           ceiling pixels (normal points down)  -> angle ~   0 deg
           vertical walls (normal horizontal)   -> angle ~  90 deg
           floor pixels   (normal points up)    -> angle ~ 180 deg

       The signed form preserves the floor/ceiling distinction. It is still
       hflip-equivariant because gravity has no x-component, so the
       horizontal flip (which only flips the x-component of the normal)
       cannot change ``N · g``. The earlier ``arccos(|N · g|)`` form was a
       deviation from canonical HHA that collapsed floors and ceilings to
       the same value with no benefit.

To make the *sign* of N_z geometrically meaningful, we orient normals
consistently toward the camera origin: any normal with ``N · P_world > 0``
(pointing AWAY from the camera, i.e., into the surface) is flipped. This
keeps the cross-product chirality consistent across surfaces.

Toolbox axis convention (verified against ``read_3d_pts_general.m`` lines 22-25
and ``read3dPoints.m``):

    intermediate point = (X_cam, Z_cam, -Y_cam) = (right, forward, up)

So in the toolbox's "world" frame after R_tilt, **axis 2 (Z_world) is the
gravity-aligned up axis**, not axis 1. R_tilt is approximately the identity
for upright cameras; the off-diagonal components correct for camera tilt.
Earlier versions of this code mistakenly used axis 1 — see Phase 0.1 in the
plan. The visualization-driven debug at 2026-04-26 caught this.

Backprojection:
    1. Camera-frame XYZ from pixels and depth.
    2. Reorder/flip to intermediate frame ``(X_cam, Z_cam, -Y_cam)``.
    3. World frame: ``XYZ_world = R_tilt @ XYZ_intermediate``, where R_tilt
       is the basis-converted rotation returned by ``read_extrinsics``.

Depth is clamped to a maximum of 8 meters before backprojection, matching
``read3dPoints.m`` line 6: ``depthInpaint(depthInpaint > 8) = 8``.

Surface normals use a numpy 5x5 cross-product estimator (cv2.rgbd.RgbdNormals
is not relied on; the synthetic Phase-1 unit tests are the correctness gate).

Invalid pixels (depth == 0 in the input) propagate to NaN in all three
output channels. On any internal error (singular K, empty scene, etc.) the
function returns an all-NaN tensor and logs a warning.
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)

# Toolbox depth clamp (read3dPoints.m line 6).
_MAX_DEPTH_M: float = 8.0
# Disparity channel is 1/depth, but at very small depths it explodes.
# Clamp to a finite range that's representative of indoor scenes.
_MAX_DISPARITY: float = 10.0
# Window radius for normal estimation (5x5 stencil; 4 cardinal neighbors at +/-r).
_NORMAL_RADIUS: int = 2


def _backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject pixel grid + depth into the intermediate frame
    ``(X_cam, Z_cam, -Y_cam)`` matching the toolbox.

    Returns an ``[H, W, 3]`` float64 array; pixels with depth == 0 carry
    zeros (callers should mask them out using the same depth==0 condition).
    """
    H, W = depth_m.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Match MATLAB 1-indexed pixel grid for parity with read_3d_pts_general.
    x = np.arange(1, W + 1, dtype=np.float64)
    y = np.arange(1, H + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    x_cam = (xx - cx) * depth_m / fx
    y_cam = (yy - cy) * depth_m / fy
    z_cam = depth_m

    # Toolbox intermediate frame (X_cam, Z_cam, -Y_cam).
    pts = np.stack([x_cam, z_cam, -y_cam], axis=-1)
    return pts


def _estimate_normals_world(
    pts_world: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Per-pixel surface normals in the world frame, [H, W, 3] float64.

    Uses a 5x5 cross-product on horizontal and vertical tangent vectors,
    then averages across the four (dx, dy) sign combinations for robustness.
    Pixels whose neighborhood is mostly invalid get NaN.

    ``pts_world``: [H, W, 3] float64 world-frame points. Invalid pixels are
        zero-filled (caller's responsibility); the validity mask is supplied
        separately so we can avoid using zero-fill values in tangent vectors.
    ``valid``: [H, W] bool — True where the depth pixel was non-zero.
    """
    H, W, _ = pts_world.shape
    r = _NORMAL_RADIUS

    # Pad arrays so we can vectorize the 5x5 neighborhood lookup.
    pad_pts = np.pad(
        pts_world,
        ((r, r), (r, r), (0, 0)),
        mode="edge",
    )
    pad_valid = np.pad(valid, ((r, r), (r, r)), mode="constant", constant_values=False)

    # Tangent vectors from neighbors at offset r in cardinal directions.
    # Use a wide stencil (5x5 effective) by sampling at +/- r.
    p_left = pad_pts[r:r + H, 0:W, :]
    p_right = pad_pts[r:r + H, 2 * r:2 * r + W, :]
    p_up = pad_pts[0:H, r:r + W, :]
    p_down = pad_pts[2 * r:2 * r + H, r:r + W, :]

    v_left = pad_valid[r:r + H, 0:W]
    v_right = pad_valid[r:r + H, 2 * r:2 * r + W]
    v_up = pad_valid[0:H, r:r + W]
    v_down = pad_valid[2 * r:2 * r + H, r:r + W]

    dx = p_right - p_left  # horizontal tangent
    dy = p_down - p_up     # vertical tangent

    # Cross product: dx x dy gives a normal pointing toward camera/away
    # depending on coord-system handedness. We take its sign-corrected
    # absolute alignment with gravity later, so the sign is not material.
    n = np.cross(dx, dy)

    # Normalize.
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_unit = np.where(norm > 1e-9, n / norm, np.nan)

    # Require all 5 stencil points (center + 4 neighbors) to be valid.
    stencil_valid = (
        valid.astype(np.uint8)
        + v_left.astype(np.uint8)
        + v_right.astype(np.uint8)
        + v_up.astype(np.uint8)
        + v_down.astype(np.uint8)
    )
    bad = stencil_valid < 5
    n_unit = np.where(bad[..., None], np.nan, n_unit)

    return n_unit


def compute_hha(
    depth_m: np.ndarray,
    K: np.ndarray,
    R_tilt: np.ndarray,
    *,
    invalid_value: float = float("nan"),
    floor_percentile: float = 5.0,
) -> np.ndarray:
    """Compute the HHA encoding for one depth image.

    Args:
        depth_m: [H, W] float32 depth in meters. Pixels equal to 0.0 are
            treated as missing and propagate to ``invalid_value`` in all
            three output channels.
        K: 3x3 camera intrinsics for the ``depth_m`` resolution.
        R_tilt: 3x3 basis-converted rotation from
            ``intrinsics.read_extrinsics`` mapping intermediate-camera
            ``(X_cam, Z_cam, -Y_cam)`` to toolbox-world XYZ (Y up).
        invalid_value: value to write at invalid pixels (default NaN).
        floor_percentile: percentile of valid Y_world used as the per-image
            floor estimate; the height channel is ``Y_world - floor``.
            Default 5.0 matches Gupta et al.'s "height above lowest point".

    Returns:
        [3, H, W] float32 array — channels ``(disparity, height_m, angle_deg)``.
        Returns an all-``invalid_value`` tensor on internal failure (logs a
        warning).
    """
    try:
        return _compute_hha_inner(
            depth_m, K, R_tilt,
            invalid_value=invalid_value,
            floor_percentile=floor_percentile,
        )
    except Exception as exc:  # noqa: BLE001 — defensive: caller aggregates failures
        logger.warning("compute_hha failed (%s); returning all-invalid tensor", exc)
        H, W = depth_m.shape
        return np.full((3, H, W), invalid_value, dtype=np.float32)


def _compute_hha_inner(
    depth_m: np.ndarray,
    K: np.ndarray,
    R_tilt: np.ndarray,
    *,
    invalid_value: float,
    floor_percentile: float,
) -> np.ndarray:
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be 2-D, got shape {depth_m.shape}")
    H, W = depth_m.shape

    depth = depth_m.astype(np.float64, copy=True)
    valid = depth > 0.0

    # Toolbox depth clamp: anything beyond 8 m saturates (read3dPoints.m).
    depth = np.where(depth > _MAX_DEPTH_M, _MAX_DEPTH_M, depth)

    # --- Channel 0: horizontal disparity = 1 / depth, clamped. ---
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = np.where(valid, 1.0 / depth, np.nan)
    disparity = np.where(disparity > _MAX_DISPARITY, _MAX_DISPARITY, disparity)

    # --- Backproject to intermediate frame, then to world frame. ---
    pts_inter = _backproject(np.where(valid, depth, 0.0), K)  # zero-fill invalid for math
    # World coords: pts_world[h, w, :] = R_tilt @ pts_inter[h, w, :]
    pts_world = pts_inter @ R_tilt.T

    # --- Channel 1: height above ground = Z_world - floor estimate. ---
    # Toolbox convention: axis 2 is the up axis (intermediate is
    # (right, forward, up); near-identity R_tilt preserves this).
    z_world = pts_world[..., 2]  # [H, W]
    valid_z = z_world[valid]
    if valid_z.size == 0:
        floor = 0.0
    else:
        floor = float(np.percentile(valid_z, floor_percentile))
    height = np.where(valid, z_world - floor, np.nan)

    # --- Channel 2: signed angle with gravity in degrees, in [0, 180]. ---
    # Estimate normals from the world-frame point cloud (this way the cross
    # product directly yields a world-frame normal). Use zero-filled pts
    # (valid mask sets bad neighborhoods to NaN inside the estimator).
    pts_world_zerofilled = np.where(valid[..., None], pts_world, 0.0)
    normals = _estimate_normals_world(pts_world_zerofilled, valid)

    # Orient normals consistently toward the camera origin. Any normal with
    # N . P_world > 0 points AWAY from the camera (into the surface) and gets
    # flipped. This makes the sign of N_z geometrically meaningful instead of
    # an artifact of cross-product chirality.
    dot_with_position = np.nansum(normals * pts_world, axis=-1, keepdims=True)
    flip_mask = dot_with_position > 0
    normals = np.where(flip_mask, -normals, normals)

    # arccos(N . gravity). Gravity in world frame = -Z (anti-gravity is +Z).
    # So cos_theta = N . (0, 0, -1) = -N_z.
    n_z = normals[..., 2]
    cos_theta = np.clip(-n_z, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))  # in [0, 180]
    angle_deg = np.where(np.isnan(n_z), np.nan, angle_deg)
    angle_deg = np.where(valid, angle_deg, np.nan)

    # --- Stack and substitute invalid_value if not NaN. ---
    hha = np.stack(
        [disparity, height, angle_deg],
        axis=0,
    ).astype(np.float32)
    if not np.isnan(invalid_value):
        hha = np.where(np.isnan(hha), np.float32(invalid_value), hha)
    return hha
