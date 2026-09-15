# save as build_reference_pcd.py somewhere (e.g. in same package)
import open3d as o3d
import numpy as np
import os

# The confidential CAD mesh is not distributed with this repository.
DRONE_STL = os.environ.get("DRONE_MESH_PATH", "")
OUTPUT_PCD = os.environ.get("DRONE_REFERENCE_PCD_OUTPUT", "drone_reference.pcd")
if not DRONE_STL:
    raise RuntimeError("Set DRONE_MESH_PATH to a locally supplied drone mesh before building a reference cloud.")

mesh = o3d.io.read_triangle_mesh(DRONE_STL)
pcd = mesh.sample_points_uniformly(number_of_points=5000)
pcd.scale(0.001, center=np.zeros(3))  # mm -> m

bbox = pcd.get_axis_aligned_bounding_box()
print("Extent (m):", np.round(bbox.get_extent(), 3))
print("Center:", np.round(bbox.get_center(), 3))

pcd.translate(-bbox.get_center())  # temporarily center at origin
o3d.io.write_point_cloud(OUTPUT_PCD, pcd)
print("Saved to", OUTPUT_PCD)
