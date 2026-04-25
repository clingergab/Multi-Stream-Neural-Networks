"""Read SUN RGB-D camera intrinsics and gravity-aligning extrinsics (R_tilt).

Conventions verified against the SUN RGB-D MATLAB toolbox
(`SUNRGBDtoolbox/readData/read3dPoints.m`,
 `SUNRGBDtoolbox/readData/read_3d_pts_general.m`,
 `SUNRGBDtoolbox/readframeSUNRGBD.m`):

- ``intrinsics.txt`` contains 9 floats, row-major, that reshape to a 3x3 ``K``.
- ``extrinsics/`` contains one or more timestamp-named ``.txt`` files with a
  3x4 raw extrinsics matrix per file. The toolbox always loads the
  **lexicographically last** file (``filename(end)`` after ``dir('*.txt')``)
  and uses only the upper-left 3x3 block.
- The raw R_tilt is then converted to the toolbox's "MATLAB world" convention:
      R_converted = M_left @ R_raw @ M_right
  with ``M_left = [[1,0,0],[0,0,1],[0,-1,0]]`` and
       ``M_right = [[1,0,0],[0,0,-1],[0,1,0]]``.
- World-frame Y is "up" after this conversion. Gravity points in -Y, anti-
  gravity in +Y. The HHA height channel is therefore the Y component of the
  world-frame point cloud (with a per-image floor offset subtracted).
"""

import os

import numpy as np


# Basis-change matrices that convert the raw extrinsics file convention into
# the toolbox's "MATLAB world" convention (see readframeSUNRGBD.m lines 46/117).
_M_LEFT = np.array(
    [[1, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    dtype=np.float64,
)

_M_RIGHT = np.array(
    [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]],
    dtype=np.float64,
)


def read_intrinsics(scene_dir: str) -> np.ndarray:
    """Read the 3x3 camera intrinsics matrix K for a SUN RGB-D scene.

    K is for the **native** depth/RGB resolution of this scene; HHA
    computation runs at native resolution against this K.
    """
    path = os.path.join(scene_dir, "intrinsics.txt")
    with open(path, "r") as f:
        vals = np.fromstring(f.read(), sep=" ", dtype=np.float64)
    if vals.size != 9:
        raise ValueError(
            f"Expected 9 floats in {path}, got {vals.size}"
        )
    return vals.reshape(3, 3)


def read_extrinsics(scene_dir: str) -> np.ndarray:
    """Read the gravity-aligning rotation R_tilt for a SUN RGB-D scene.

    Selects the lexicographically last ``.txt`` file in ``extrinsics/``
    (matching ``filename(end)`` after ``dir('*.txt')`` in the toolbox),
    reads its upper-left 3x3 block as the raw R_tilt, and applies the
    toolbox's basis conversion. The returned matrix maps a point in the
    intermediate camera frame ``(X_cam, Z_cam, -Y_cam)`` to the toolbox
    world frame ``(X_world, Y_world, Z_world)`` with Y as the up axis.
    """
    extr_dir = os.path.join(scene_dir, "extrinsics")
    if not os.path.isdir(extr_dir):
        raise FileNotFoundError(f"No extrinsics directory at {extr_dir}")

    files = sorted(f for f in os.listdir(extr_dir) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {extr_dir}")

    path = os.path.join(extr_dir, files[-1])
    with open(path, "r") as f:
        vals = np.fromstring(f.read(), sep=" ", dtype=np.float64)
    # Files have either 9 (3x3) or 12 (3x4) floats; toolbox uses (1:3, 1:3).
    if vals.size == 12:
        raw = vals.reshape(3, 4)[:, :3]
    elif vals.size == 9:
        raw = vals.reshape(3, 3)
    else:
        raise ValueError(
            f"Unexpected float count in {path}: got {vals.size}, expected 9 or 12"
        )

    return _M_LEFT @ raw @ _M_RIGHT
