#!/usr/bin/env python3
"""
Diagnostic utility: loads the local drone STL, prints bounding box extents
and axis orientations for coordinate alignment verification.
"""
import open3d as o3d
import numpy as np
import os

def inspect_stl(path, scale=0.001):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.scale(scale, center=np.zeros(3))
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    center = bbox.get_center()
    print(f"\n{'='*50}")
    print(f"File: {path}")
    print(f"Loaded mesh bounding box (local units): Extent X={extent[0]:.3f}m  Y={extent[1]:.3f}m  Z={extent[2]:.3f}m")
    print(f"  Center: X={center[0]:.3f}m  Y={center[1]:.3f}m  Z={center[2]:.3f}m")
    print(f"  → Longest axis: {'X' if extent[0]>extent[1] and extent[0]>extent[2] else 'Y' if extent[1]>extent[2] else 'Z'}")
    print(f"  → Thinnest axis (should be UP after rotation): {'Z' if extent[2]<extent[0] and extent[2]<extent[1] else 'Y' if extent[1]<extent[0] else 'X'}")

# The confidential CAD mesh is not distributed with this repository.
local_mesh = os.environ.get("DRONE_MESH_PATH", "")
if not local_mesh:
    raise RuntimeError("Set DRONE_MESH_PATH to a locally supplied drone mesh before running this diagnostic.")
inspect_stl(local_mesh)
