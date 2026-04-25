"""HHA (Horizontal disparity, Height above ground, Angle with gravity) encoding.

Converts SUN RGB-D depth images into 3-channel HHA representations using the
canonical SUN RGB-D toolbox conventions for camera intrinsics, gravity-aligning
extrinsics (R_tilt), and depth clamping.
"""

from .intrinsics import read_intrinsics, read_extrinsics
from .hha import compute_hha

__all__ = [
    "read_intrinsics",
    "read_extrinsics",
    "compute_hha",
]
