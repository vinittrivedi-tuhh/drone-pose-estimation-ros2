#!/usr/bin/env python3
"""
Pure ICP Drone Pose Estimator Node (Zero ArUco Dependency).
Performs markerless initialization via workspace crop, dynamic Z-peak passband filtering,
DBSCAN clustering, and a fine-pass multi-start yaw search to resolve 180° orientation ambiguity.
Transitions to high-speed (1-5 ms) Point-to-Plane temporal tracking.
"""

import time
import os
import copy
from typing import Optional, Tuple, List
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2

from drone_pose_estimation.common import CameraConfig, resolve_mesh_path, SharedCloudProcessor

class PureConfig:
    # STL Path (resolved dynamically via package share or search path)
    stl_path: str = resolve_mesh_path()
    reference_n_points: int = 10_000
    stl_scale: float = 0.001

    # Workspace Limits in camera frame (Global Crop ROI)
    # Define a bounding box in camera coordinates to isolate the region of interest.
    workspace_x_min: float = -1.5
    workspace_x_max: float = 1.5
    workspace_y_min: float = -1.5
    workspace_y_max: float = 1.5
    workspace_z_min: float = 0.3
    workspace_z_max: float = 2.2

    # Dynamic Bounding Box Crop sizes (half extent in meters) for Tracking mode
    crop_half_extent_track: float = 1.2   # Keep crop wide to prevent cutting off drone geometry during tracking

    # Mounting structure / mounting structure carve-out bounds in camera frame (x_min, x_max, y_min, y_max).
    # These describe the physical support column footprint in camera coordinates (confidential platform asset).
    # Set CARVE_BOUNDS="x_min,x_max,y_min,y_max" to enable local structure carving.
    # By default None: no carving occurs until explicitly configured locally.
    carve_bounds_env = os.environ.get("CARVE_BOUNDS", None)
    carve_bounds: Optional[Tuple[float, float, float, float]] = (
        tuple(float(v) for v in carve_bounds_env.split(",")) if carve_bounds_env else None
    )

    # Voxel downsampling size
    voxel_size: float = 0.01

    # DBSCAN parameters
    dbscan_eps: float = 0.03
    dbscan_min_pts: int = 10
    min_cluster_points: int = 200

    # ICP parameters
    icp_max_dist_coarse: float = 0.15  # Pass 1 coarse search (m)
    icp_max_dist_fine: float = 0.05   # Pass 2 fine alignment (m)
    icp_max_iterations_coarse: int = 30  # Fast limit for coarse search
    icp_max_iterations_fine: int = 100   # Standard limit for fine pass

    # Quality thresholds for accepting ICP result
    min_fitness: float = 0.05         # Minimum fraction of points matching (lowered to 0.05 to handle 10k Poisson points vs sparse target)
    max_rmse: float = 0.05            # Max acceptable RMSE of matched pairs (5cm)
    max_rmse_loose: float = 0.10

    # Top-face filtering threshold
    top_face_normal_z_threshold: float = 0.0

    # Drone dimensions
    drone_half_height: float = 0.28   # m

    # Timing
    timer_period_sec: float = 0.1     # 10 Hz nominal loop rate

    # Candidate yaw angles for multi-start coarse alignment during initialization
    yaw_candidates_deg: List[float] = [0.0, 90.0, 180.0, 270.0]


class PureSTLLoader:
    """Loads and preprocesses CAD mesh into a reference point cloud for Pure ICP."""
    def __init__(self, cfg: PureConfig, logger):
        self._cfg = cfg
        self._log = logger
        self.reference_pcd: Optional[o3d.geometry.PointCloud] = None
        self.reference_down: Optional[o3d.geometry.PointCloud] = None

    def load(self) -> bool:
        # Resolve path dynamically if it doesn't exist (makes it portable)
        path = self._cfg.stl_path
        if not path or not os.path.exists(path):
            self._log.error(
                "No local mesh is configured. The confidential CAD asset is not "
                "included; set the stl_path ROS parameter or DRONE_MESH_PATH."
            )
            return False

        self._log.info(f"[STLLoader] Loading STL: {path}")
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh.is_empty():
                self._log.error(f"[STLLoader] Mesh is empty: {path}")
                return False

            mesh.scale(self._cfg.stl_scale, center=np.zeros(3))
            center = mesh.get_center()
            mesh.translate(-center)
            mesh.compute_triangle_normals()

            tri_normals = np.asarray(mesh.triangle_normals)
            top_mask = tri_normals[:, 2] > self._cfg.top_face_normal_z_threshold
            top_indices = np.where(top_mask)[0]

            mesh_top = mesh.select_by_index(top_indices)
            self._log.info(
                f"[STLLoader] Filtered to top-facing surfaces: "
                f"{len(top_indices):,}/{len(tri_normals):,} triangles"
            )

            pcd = mesh_top.sample_points_poisson_disk(
                number_of_points=self._cfg.reference_n_points,
                init_factor=5
            )
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.03, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            self.reference_pcd = pcd
            self.reference_down = pcd.voxel_down_sample(self._cfg.voxel_size)

            extent = pcd.get_axis_aligned_bounding_box().get_extent()
            self._log.info(
                f"[STLLoader] Reference PCD ready: {len(pcd.points):,} pts "
                f"(downsampled: {len(self.reference_down.points):,} pts, "
                f"extent: {extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f} m)"
            )
            return True

        except Exception as exc:
            self._log.error(f"[STLLoader] Failed to load STL: {exc}")
            return False


class PureCloudProcessor:
    """Processes incoming PointCloud2 for Pure ICP."""
    def __init__(self, cfg: PureConfig, logger):
        self._cfg = cfg
        self._log = logger

    def ros_to_open3d(self, msg: PointCloud2) -> o3d.geometry.PointCloud:
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        valid = ~np.isnan(pts).any(axis=1)
        pts = pts[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pcd

    def global_crop_and_filter(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Crop to overall workspace ROI, optionally carve configured structures, and passband filter.
        """
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([self._cfg.workspace_x_min, self._cfg.workspace_y_min, self._cfg.workspace_z_min]),
            max_bound=np.array([self._cfg.workspace_x_max, self._cfg.workspace_y_max, self._cfg.workspace_z_max])
        )
        cropped = pcd.crop(bbox)
        
        if len(cropped.points) == 0:
            return cropped

        # Optional: Carve out mounting structure in camera frame if configured
        pts = np.asarray(cropped.points)
        if self._cfg.carve_bounds is not None:
            x_min, x_max, y_min, y_max = self._cfg.carve_bounds
            x = pts[:, 0]
            y = pts[:, 1]
            is_carved = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            pts_filtered = pts[~is_carved]
        else:
            pts_filtered = pts

        if len(pts_filtered) == 0:
            return o3d.geometry.PointCloud()

        # Dynamic Z-peak passband filter:
        z_peak = np.percentile(pts_filtered[:, 2], 1.0)
        
        valid_mask = (pts_filtered[:, 2] >= z_peak) & (pts_filtered[:, 2] < z_peak + 0.35)
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(pts_filtered[valid_mask])
        
        self._log.debug(
            f"[PureCloudProcessor] Crop & Z-Peak Filter: {len(pcd.points):,} -> {len(filtered_pcd.points):,} points "
            f"(clutter removed: {len(cropped.points) - len(filtered_pcd.points)} pts)"
        )
        return filtered_pcd

    def local_crop_track(self, pcd: o3d.geometry.PointCloud, center: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Crop a tight bounding box around the previous estimated center during tracking.
        """
        h = self._cfg.crop_half_extent_track
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=center - h, max_bound=center + h)
        cropped = pcd.crop(bbox)

        if len(cropped.points) == 0:
            return cropped

        pts = np.asarray(cropped.points)
        z_max_threshold = center[2] + self._cfg.drone_half_height
        valid_mask = pts[:, 2] < z_max_threshold
        filtered_pts = pts[valid_mask]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_pts)
        return filtered_pcd

    def voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(voxel_size=self._cfg.voxel_size)

    def isolate_drone_cluster(self, pcd: o3d.geometry.PointCloud) -> Tuple[Optional[o3d.geometry.PointCloud], int]:
        """
        Run DBSCAN clustering and extract the largest cluster corresponding to the drone body.
        """
        if len(pcd.points) < self._cfg.min_cluster_points:
            return None, 0

        labels = np.array(
            pcd.cluster_dbscan(
                eps=self._cfg.dbscan_eps,
                min_points=self._cfg.dbscan_min_pts,
                print_progress=False
            )
        )

        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return None, 0

        best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        drone_pts = np.asarray(pcd.points)[labels == best_label]

        if len(drone_pts) < self._cfg.min_cluster_points:
            return None, 0

        drone_pcd = o3d.geometry.PointCloud()
        drone_pcd.points = o3d.utility.Vector3dVector(drone_pts)
        return drone_pcd, len(drone_pts)


class PureICPEstimatorNode(Node):
    """
    ROS 2 Node for Pure ICP (Markerless) 6-DoF Drone Pose Estimation.
    """
    # Dynamic offsets configured locally
    OFFSET_MESH_BASE_LINK = CameraConfig.OFFSET_MESH_BASE_LINK

    def __init__(self):
        super().__init__("pure_icp_pose_estimator")
        self._cfg = PureConfig()
        
        # ROS Parameters
        self.declare_parameter("cam_pitch_deg", -90.0)
        self.declare_parameter("cam_roll_deg", 0.0)
        self.declare_parameter("cam_yaw_deg", 180.0)
        self.declare_parameter("stl_path", "")

        stl_param = self.get_parameter("stl_path").get_parameter_value().string_value
        if stl_param:
            self._cfg.stl_path = stl_param

        # State machine
        self._state = "INITIALIZING"   # "INITIALIZING" | "TRACKING" | "LOST"
        self._latest_pointcloud: Optional[PointCloud2] = None
        self._previous_T_cam_drone: Optional[np.ndarray] = None
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 5

        # Utilities
        self._loader = PureSTLLoader(self._cfg, self.get_logger())
        self._processor = PureCloudProcessor(self._cfg, self.get_logger())

        if not self._loader.load():
            self.get_logger().error("Pure ICP failed to initialize CAD reference model. Node stopping.")
            return

        # ROS Comm
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self._sub_pcl = self.create_subscription(
            PointCloud2,
            "/depth_pcl",
            self._pointcloud_callback,
            qos
        )

        self._pub_pose = self.create_publisher(PoseStamped, "/drone_pose_icp_pure", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._timer = self.create_timer(self._cfg.timer_period_sec, self._estimation_loop)
        self.get_logger().info("Pure ICP Pose Estimator initialized (Zero ArUco Mode).")

    def _pointcloud_callback(self, msg: PointCloud2):
        self._latest_pointcloud = msg

    def _estimation_loop(self):
        if self._latest_pointcloud is None:
            return

        t0 = time.perf_counter()
        raw_pcd = self._processor.ros_to_open3d(self._latest_pointcloud)

        if self._state in ["INITIALIZING", "LOST"]:
            success, T_cam_drone = self._run_markerless_initialization(raw_pcd)
        else:
            success, T_cam_drone = self._run_tracking_step(raw_pcd)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if success and T_cam_drone is not None:
            self._previous_T_cam_drone = T_cam_drone
            self._consecutive_failures = 0
            self._state = "TRACKING"
            self._publish_pose(T_cam_drone, elapsed_ms)
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self._state == "TRACKING":
                    self.get_logger().warn("[PureICP] Lost tracking lock. Re-entering INITIALIZING state.")
                self._state = "LOST"

    def _run_markerless_initialization(self, raw_pcd: o3d.geometry.PointCloud) -> Tuple[bool, Optional[np.ndarray]]:
        """Markerless init with fine-pass multi-start yaw evaluation."""
        cropped = self._processor.global_crop_and_filter(raw_pcd)
        down = self._processor.voxel_downsample(cropped)
        cluster, n_pts = self._processor.isolate_drone_cluster(down)

        if cluster is None:
            return False, None

        cluster_pts = np.asarray(cluster.points)
        centroid = np.mean(cluster_pts, axis=0)

        best_T = None
        best_fitness = -1.0
        best_rmse = 999.0

        for yaw_deg in self._cfg.yaw_candidates_deg:
            R_cand = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
            T_init = np.eye(4)
            T_init[:3, :3] = R_cand
            T_init[:3, 3] = centroid

            res_coarse = self._run_icp_alignment(
                self._loader.reference_down,
                cluster,
                T_init,
                max_corr_dist=self._cfg.icp_max_dist_coarse,
                is_fine_only=False
            )

            res_fine = self._run_icp_alignment(
                self._loader.reference_pcd,
                cluster,
                res_coarse.transformation,
                max_corr_dist=self._cfg.icp_max_dist_fine,
                is_fine_only=True
            )

            if res_fine.fitness > best_fitness:
                best_fitness = res_fine.fitness
                best_rmse = res_fine.inlier_rmse
                best_T = res_fine.transformation

        if best_fitness >= self._cfg.min_fitness and best_rmse <= self._cfg.max_rmse_loose:
            self.get_logger().info(
                f"[PureICP - Init] Lock Established! Best fitness: {best_fitness:.3f}, RMSE: {best_rmse*100:.2f} cm"
            )
            return True, best_T

        return False, None

    def _run_tracking_step(self, raw_pcd: o3d.geometry.PointCloud) -> Tuple[bool, Optional[np.ndarray]]:
        """Fast (1-5 ms) temporal tracking pass using previous frame transformation."""
        prev_center = self._previous_T_cam_drone[:3, 3]
        cropped = self._processor.local_crop_track(raw_pcd, prev_center)
        down = self._processor.voxel_downsample(cropped)

        if len(down.points) < self._cfg.min_cluster_points:
            return False, None

        # Point-to-Plane Fine ICP alignment
        res = self._run_icp_alignment(
            self._loader.reference_pcd,
            down,
            self._previous_T_cam_drone,
            max_corr_dist=self._cfg.icp_max_dist_fine,
            is_fine_only=True
        )

        if res.fitness >= self._cfg.min_fitness and res.inlier_rmse <= self._cfg.max_rmse:
            return True, res.transformation

        return False, None

    def _run_icp_alignment(
        self,
        ref_pcd: o3d.geometry.PointCloud,
        live_pcd: o3d.geometry.PointCloud,
        T_init: np.ndarray,
        max_corr_dist: float,
        is_fine_only: bool = False
    ) -> o3d.pipelines.registration.RegistrationResult:
        if not is_fine_only:
            return o3d.pipelines.registration.registration_icp(
                ref_pcd, live_pcd,
                max_corr_dist, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._cfg.icp_max_iterations_coarse)
            )

        if not live_pcd.has_normals():
            live_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )

        return o3d.pipelines.registration.registration_icp(
            ref_pcd, live_pcd,
            max_corr_dist, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._cfg.icp_max_iterations_fine)
        )

    def _publish_pose(self, T_final: np.ndarray, elapsed_ms: float):
        # Base link translation and rotation in camera frame
        R_mesh = T_final[:3, :3]
        t_base_link = T_final[:3, 3] + R_mesh @ self.OFFSET_MESH_BASE_LINK
        
        q_mesh = Rotation.from_matrix(R_mesh).as_quat()

        now = self.get_clock().now().to_msg()

        msg = PoseStamped()
        msg.header.stamp = now
        msg.header.frame_id = "sim_camera"

        msg.pose.position.x = float(t_base_link[0])
        msg.pose.position.y = float(t_base_link[1])
        msg.pose.position.z = float(t_base_link[2])
        msg.pose.orientation.x = float(q_mesh[0])
        msg.pose.orientation.y = float(q_mesh[1])
        msg.pose.orientation.z = float(q_mesh[2])
        msg.pose.orientation.w = float(q_mesh[3])

        self._pub_pose.publish(msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "sim_camera"
        tf_msg.child_frame_id = "drone_base_link_icp_pure"
        tf_msg.transform.translation.x = float(t_base_link[0])
        tf_msg.transform.translation.y = float(t_base_link[1])
        tf_msg.transform.translation.z = float(t_base_link[2])
        tf_msg.transform.rotation = msg.pose.orientation
        self._tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PureICPEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
