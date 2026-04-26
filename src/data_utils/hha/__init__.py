"""HHA (Horizontal disparity, Height above ground, Angle with gravity) encoding.

Converts SUN RGB-D depth images into 3-channel HHA representations using the
canonical SUN RGB-D toolbox conventions for camera intrinsics, gravity-aligning
extrinsics (R_tilt), and depth clamping.
"""

from .intrinsics import read_intrinsics, read_extrinsics
from .hha import compute_hha
from .scannet_intrinsics import (
    ScannetSceneMeta,
    angular_distance_deg,
    compute_scannet_rtilt,
    load_drop_list,
    read_axis_alignment,
    read_intrinsic_depth,
    read_pose,
    should_drop,
)

__all__ = [
    # SUN RGB-D toolbox
    "read_intrinsics",
    "read_extrinsics",
    # ScanNet helpers
    "ScannetSceneMeta",
    "angular_distance_deg",
    "compute_scannet_rtilt",
    "load_drop_list",
    "read_axis_alignment",
    "read_intrinsic_depth",
    "read_pose",
    "should_drop",
    # Core HHA
    "compute_hha",
]
