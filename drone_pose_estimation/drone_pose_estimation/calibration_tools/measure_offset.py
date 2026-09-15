#!/usr/bin/env python3
"""
Run this ONCE to measure T_MODEL_TO_MARKER from live data.
It subscribes to /drone_pose_aruco and /depth_pcl,
finds the largest cluster centroid, and prints the offset.

Usage:
  python3 measure_offset.py
  
Let it run for ~10 frames, then Ctrl+C.
Average the printed offsets and paste into Config.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d

DBSCAN_EPS      = 0.05
DBSCAN_MIN_PTS  = 5
VOXEL_SIZE      = 0.03
CROP_HALF       = 1.5     # metres around ArUco position

class OffsetMeasurer(Node):
    def __init__(self):
        super().__init__("offset_measurer")
        self._aruco = None
        self._pcl   = None
        self._frame = 0

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(PoseStamped, "/drone_pose_aruco", self._a_cb, qos)
        self.create_subscription(PointCloud2,  "/depth_pcl",        self._p_cb, qos)
        self.create_timer(0.5, self._run)
        self.get_logger().info("Measuring T_MODEL_TO_MARKER — let run for 10 frames then Ctrl+C")

    def _a_cb(self, msg): self._aruco = msg
    def _p_cb(self, msg): self._pcl   = msg

    def _run(self):
        if self._aruco is None or self._pcl is None:
            self.get_logger().warn("Waiting for messages…")
            return

        # ArUco position in camera frame
        p = self._aruco.pose.position
        aruco_pos = np.array([p.x, p.y, p.z])

        # Build live cloud
        pts_raw = np.array(list(pc2.read_points(
            self._pcl, field_names=("x","y","z"), skip_nans=True)))
        pts = np.column_stack([pts_raw["x"], pts_raw["y"], pts_raw["z"]]).astype(np.float32)
        # Convert from Isaac Sim coordinates (meters) to Open3D PCD
        if pts.shape[0] == 0:
            self.get_logger().warn("Empty point cloud"); return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:,:3].astype(np.float64))

        # Crop around ArUco marker
        h = CROP_HALF
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=aruco_pos - h, max_bound=aruco_pos + h)
        pcd = pcd.crop(bbox)

        if len(pcd.points) < 50:
            self.get_logger().warn(f"Only {len(pcd.points)} pts in crop — check ArUco Z or crop size")
            return

        # Voxel downsample
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)

        # DBSCAN
        labels = np.array(pcd.cluster_dbscan(
            eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS, print_progress=False))
        unique = set(labels) - {-1}
        if not unique:
            self.get_logger().warn("No clusters found"); return

        # Largest cluster centroid
        best = max(unique, key=lambda l: np.sum(labels==l))
        cluster_pts = np.asarray(pcd.points)[labels == best]
        centroid = cluster_pts.mean(axis=0)

        # THE OFFSET
        offset = centroid - aruco_pos

        self._frame += 1
        print(f"\n{'='*55}")
        print(f"  Frame {self._frame}")
        print(f"  ArUco pos  : [{aruco_pos[0]:+.4f}, {aruco_pos[1]:+.4f}, {aruco_pos[2]:+.4f}]")
        print(f"  Centroid   : [{centroid[0]:+.4f}, {centroid[1]:+.4f}, {centroid[2]:+.4f}]")
        print(f"  OFFSET (centroid - aruco):")
        print(f"    DX = {offset[0]:+.4f} m")
        print(f"    DY = {offset[1]:+.4f} m")
        print(f"    DZ = {offset[2]:+.4f} m")
        print(f"  Cluster pts: {len(cluster_pts):,}")
        print(f"{'='*55}")
        print(f"\n  ➜ Paste into Config:")
        print(f"    self.T_MODEL_TO_MARKER = np.array([{offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f}])")

def main():
    rclpy.init()
    node = OffsetMeasurer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()