"""
Shared Configuration, Geometry Utilities, and Preprocessing for Drone Pose Estimation.
Provides dynamic path resolution, camera coordinate conversions, CAD STL loading, and point cloud filtering.
"""

import os
import re
import numpy as np
import open3d as o3d
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


class CameraConfig:
    """
    Centralized calibration constants and coordinate frame transformations.
    """
    # Camera world position in ROS Z-up frame
    cam_pos_env = os.environ.get("CAMERA_WORLD_POS", "0.0,-1.22141,2.35389")
    CAM_WORLD_POS = np.array([float(x) for x in cam_pos_env.split(",")])

    # Nominal Camera orientation matrix in ROS Z-up world (q = [0.7071, 0.7071, 0, 0])
    R_CAM_NOMINAL = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0]
    ])

    # USD Y-up to ROS Z-up coordinate conversion matrix (x_ros=x_usd, y_ros=-z_usd, z_ros=y_usd)
    T_USD_TO_ROS = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0]
    ])

    # Reference frame alignment mapping pre-rotated reference PCD to drone body frame in camera space (0° error)
    R_REF_DRONE = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ])

    # Calibration offset specific to the confidential drone CAD and physical marker placement — not included; measure and set this yourself.
    T_MODEL_TO_MARKER = np.array([
        float(x) for x in os.environ.get("MODEL_TO_MARKER_OFFSET", "0,0,0").split(",")
    ])

    # Calibrated offsets between frames (default to [0,0,0]; configure locally for your setup)
    OFFSET_ARUCO_BASE_LINK = np.array([
        float(x) for x in os.environ.get("ARUCO_TO_BASE_LINK_OFFSET", "0,0,0").split(",")
    ])
    OFFSET_MESH_ARUCO = np.array([
        float(x) for x in os.environ.get("MESH_TO_ARUCO_OFFSET", "0,0,0").split(",")
    ])
    OFFSET_MESH_BASE_LINK = np.array([
        float(x) for x in os.environ.get("MESH_TO_BASE_LINK_OFFSET", "0,0,0").split(",")
    ])

    # ArUco marker coordinate frame to CAD reference frame basis transformation
    R_REF_MARKER_CONST = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ])

    @staticmethod
    def get_camera_world_rotation(pitch_deg: float = -90.0, roll_deg: float = 0.0, yaw_deg: float = 180.0) -> np.ndarray:
        """
        Computes active camera rotation matrix in ROS world for arbitrary tilt angles.
        """
        delta_pitch = pitch_deg - (-90.0)
        delta_roll = roll_deg - 0.0
        delta_yaw = yaw_deg - 180.0
        if abs(delta_pitch) > 1e-3 or abs(delta_roll) > 1e-3 or abs(delta_yaw) > 1e-3:
            r_delta = Rotation.from_euler('xyz', [delta_pitch, delta_roll, delta_yaw], degrees=True)
            return CameraConfig.R_CAM_NOMINAL @ r_delta.as_matrix()
        return CameraConfig.R_CAM_NOMINAL.copy()


def resolve_mesh_path(requested_path: Optional[str] = None) -> str:
    """Resolve a locally supplied confidential drone mesh, if available.

    The mesh is intentionally not distributed with this public repository.
    Supply it with the ``stl_path`` ROS parameter or the ``DRONE_MESH_PATH``
    environment variable.
    """
    if requested_path and os.path.exists(requested_path):
        return requested_path

    local_mesh_path = os.environ.get("DRONE_MESH_PATH", "")
    if local_mesh_path and os.path.exists(local_mesh_path):
        return os.path.abspath(local_mesh_path)

    return requested_path or local_mesh_path


def resolve_results_directory(subdir: str = "csv") -> str:
    """
    Dynamically finds the results output directory in the active repository without hardcoded usernames.
    """
    workspace_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        os.getcwd()
    ]
    for root in workspace_candidates:
        candidate = os.path.join(root, "new_results", subdir)
        os.makedirs(candidate, exist_ok=True)
        return candidate

    fallback = os.path.join(os.getcwd(), "results", subdir)
    os.makedirs(fallback, exist_ok=True)
    return fallback


class SharedSTLLoader:
    """
    Loads drone CAD STL, filters normals, and generates reference Open3D point clouds.
    """
    def __init__(self, stl_path: str, logger=None):
        self.stl_path = resolve_mesh_path(stl_path)
        self._logger = logger
        self.reference_pcd: Optional[o3d.geometry.PointCloud] = None
        self.reference_down: Optional[o3d.geometry.PointCloud] = None

    def log_info(self, msg: str):
        if self._logger:
            self._logger.info(msg)
        else:
            print(msg)

    def load(self, num_points: int = 10000, voxel_size: float = 0.015, z_normal_threshold: float = 0.2) -> bool:
        self.log_info(f"[STLLoader] Loading CAD STL: {self.stl_path}")
        if not os.path.exists(self.stl_path):
            if self._logger:
                self._logger.error(f"[STLLoader] File not found: {self.stl_path}")
            return False

        try:
            mesh = o3d.io.read_triangle_mesh(self.stl_path)
            if mesh.is_empty():
                return False

            # Center mesh at origin
            center = mesh.get_center()
            mesh.translate(-center)
            mesh.compute_triangle_normals()

            # Filter top-facing surfaces (visible from overhead camera)
            tri_normals = np.asarray(mesh.triangle_normals)
            top_facing_mask = tri_normals[:, 2] > z_normal_threshold
            top_facing_indices = np.where(top_facing_mask)[0]

            mesh_top = mesh.select_by_index(top_facing_indices)

            # Poisson disk sample surface points
            pcd = mesh_top.sample_points_poisson_disk(
                number_of_points=num_points,
                init_factor=5
            )
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            self.reference_pcd = pcd
            self.reference_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            self.log_info(f"[STLLoader] Reference PCD created with {len(pcd.points)} points (downsampled: {len(self.reference_down.points)} pts)")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"[STLLoader] Load error: {e}")
            return False


class SharedCloudProcessor:
    """
    Common point cloud utilities: ROS to Open3D conversion, voxel downsampling, and cluster isolation.
    """
    def __init__(self, logger=None):
        self._logger = logger

    @staticmethod
    def ros_to_open3d(ros_msg: PointCloud2) -> o3d.geometry.PointCloud:
        points_list = []
        for p in pc2.read_points(ros_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([p[0], p[1], p[2]])

        pcd = o3d.geometry.PointCloud()
        if points_list:
            pcd.points = o3d.utility.Vector3dVector(np.array(points_list, dtype=np.float64))
        return pcd

    @staticmethod
    def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
        if len(pcd.points) == 0:
            return pcd
        down = pcd.voxel_down_sample(voxel_size=voxel_size)
        down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        return down
