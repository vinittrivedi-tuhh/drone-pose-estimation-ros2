#!/usr/bin/env python3
"""
Diagnostic Node: Compares ArUco and ICP pose estimations against Simulation Ground Truth.
Prints live errors and 4x4 Transformation Matrices to the terminal.
"""

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
import tf2_ros

from drone_pose_estimation.common import CameraConfig

# ANSI terminal formatting
CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def format_matrix(T: np.ndarray, indent: str = "    ") -> str:
    lines = []
    for row in T:
        lines.append(indent + " ".join(f"{v: 8.4f}" for v in row))
    return "\n".join(lines)


class CompareErrorsNode(Node):
    OFFSET_ARUCO_BASE_LINK = CameraConfig.OFFSET_ARUCO_BASE_LINK
    R_REF_MARKER_CONST = CameraConfig.R_REF_MARKER_CONST
    R_REF_DRONE = CameraConfig.R_REF_DRONE

    def __init__(self):
        super().__init__("compare_errors_node")

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        self._latest_aruco = None
        self._latest_icp = None
        self._lock = threading.Lock()

        self.create_subscription(PoseStamped, "/drone_pose_aruco", self._aruco_cb, qos)
        self.create_subscription(PoseStamped, "/drone_pose_icp_advanced", self._icp_cb, qos)

        # 10 Hz dashboard refresh
        self.create_timer(0.1, self._update_dashboard)
        self.get_logger().info("Compare Errors Dashboard Node initialized.")

    def _aruco_cb(self, msg: PoseStamped):
        with self._lock:
            self._latest_aruco = msg

    def _icp_cb(self, msg: PoseStamped):
        with self._lock:
            self._latest_icp = msg

    def _update_dashboard(self):
        with self._lock:
            aruco_msg = self._latest_aruco
            icp_msg = self._latest_icp

        try:
            trans = self._tf_buffer.lookup_transform("world", "drone", rclpy.time.Time())

            p_usd = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            q_usd = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])

            T_u2r = CameraConfig.T_USD_TO_ROS
            p_world_ros = T_u2r @ p_usd
            R_drone_ros_world = T_u2r @ Rotation.from_quat(q_usd).as_matrix()

            cam_pos = CameraConfig.CAM_WORLD_POS
            R_cam = CameraConfig.R_CAM_NOMINAL

            # Project to camera frame
            p_gt_ros = R_cam.T @ (p_world_ros - cam_pos)
            R_gt_ros = R_cam.T @ R_drone_ros_world
            yaw_gt = Rotation.from_matrix(R_gt_ros).as_euler('xyz', degrees=True)[2]
        except Exception:
            print(f"{CLEAR}{YELLOW}Waiting for transform world -> drone on /tf...{RESET}")
            return

        aruco_pos_bl = None
        aruco_err = None
        yaw_aruco = None
        if aruco_msg is not None:
            p_ar = np.array([aruco_msg.pose.position.x, aruco_msg.pose.position.y, aruco_msg.pose.position.z])
            q_ar = [
                aruco_msg.pose.orientation.x,
                aruco_msg.pose.orientation.y,
                aruco_msg.pose.orientation.z,
                aruco_msg.pose.orientation.w
            ]
            R_ar = Rotation.from_quat(q_ar).as_matrix()
            
            aruco_pos_bl = p_ar + R_ar @ self.OFFSET_ARUCO_BASE_LINK
            aruco_err = np.linalg.norm(aruco_pos_bl - p_gt_ros)
            
            R_bl_ar = R_ar @ self.R_REF_MARKER_CONST @ self.R_REF_DRONE
            yaw_aruco = Rotation.from_matrix(R_bl_ar).as_euler('xyz', degrees=True)[2]

        icp_pos_bl = None
        icp_err = None
        yaw_icp = None
        if icp_msg is not None:
            icp_pos_bl = np.array([icp_msg.pose.position.x, icp_msg.pose.position.y, icp_msg.pose.position.z])
            icp_err = np.linalg.norm(icp_pos_bl - p_gt_ros)
            q_icp = [
                icp_msg.pose.orientation.x,
                icp_msg.pose.orientation.y,
                icp_msg.pose.orientation.z,
                icp_msg.pose.orientation.w
            ]
            yaw_icp = Rotation.from_quat(q_icp).as_euler('xyz', degrees=True)[2]

        # 4. Render Terminal Dashboard
        print(CLEAR)
        print(f"==========================================================================")
        print(f" {BOLD}{CYAN}Drone Pose Estimation Live Evaluation Dashboard{RESET} ")
        print(f"==========================================================================")
        print(f" {BOLD}{YELLOW}ACTIVE CONFIGURATION OFFSETS:{RESET}")
        print(f"  • {BOLD}ArUco Marker → base_link offset vector:{RESET} {self.OFFSET_ARUCO_BASE_LINK}")
        print(f"==========================================================================")
        print(f" {BOLD}{CYAN}LIVE COMPARISON FOR DRONE base_link ORIGIN:{RESET}")
        print(f"==========================================================================")
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt_ros
        T_gt[:3, 3] = p_gt_ros

        print(f" {BOLD}Ground Truth (base_link in sim_camera Frame):{RESET}")
        print(f"  Position:   X={p_gt_ros[0]: 8.3f} m  | Y={p_gt_ros[1]: 8.3f} m  | Z={p_gt_ros[2]: 8.3f} m")
        print(f"  Yaw Angle:  {yaw_gt: 8.1f}°")
        print(f"  {BOLD}4x4 Homogeneous Transformation Matrix (T_GT):{RESET}")
        print(format_matrix(T_gt))
        print(f"--------------------------------------------------------------------------")

        if icp_pos_bl is not None:
            T_icp = np.eye(4)
            T_icp[:3, :3] = Rotation.from_quat(q_icp).as_matrix()
            T_icp[:3, 3] = icp_pos_bl

            icp_col = GREEN if icp_err < 0.02 else YELLOW
            print(f" {BOLD}ICP Estimation (on /drone_pose_icp_advanced):{RESET}")
            print(f"  Position:   X={icp_pos_bl[0]: 8.3f} m  | Y={icp_pos_bl[1]: 8.3f} m  | Z={icp_pos_bl[2]: 8.3f} m")
            print(f"  Yaw Angle:  {yaw_icp: 8.1f}°")
            print(f"  {BOLD}4x4 Homogeneous Transformation Matrix (T_ICP):{RESET}")
            print(format_matrix(T_icp))
            print(f"  {BOLD}Absolute 3D Error:{RESET}  {icp_col}{icp_err*100: 7.2f} cm{RESET}")
        else:
            print(f" {BOLD}ICP Estimation:{RESET} {RED}Waiting for data...{RESET}")
        print(f"--------------------------------------------------------------------------")

        if aruco_pos_bl is not None:
            T_aruco = np.eye(4)
            T_aruco[:3, :3] = R_bl_ar
            T_aruco[:3, 3] = aruco_pos_bl

            aruco_col = GREEN if aruco_err < 0.02 else YELLOW
            print(f" {BOLD}ArUco Estimation (on /drone_pose_aruco + offset):{RESET}")
            print(f"  Position:   X={aruco_pos_bl[0]: 8.3f} m  | Y={aruco_pos_bl[1]: 8.3f} m  | Z={aruco_pos_bl[2]: 8.3f} m")
            print(f"  Yaw Angle:  {yaw_aruco: 8.1f}°")
            print(f"  {BOLD}4x4 Homogeneous Transformation Matrix (T_ArUco):{RESET}")
            print(format_matrix(T_aruco))
            print(f"  {BOLD}Absolute 3D Error:{RESET}  {aruco_col}{aruco_err*100: 7.2f} cm{RESET}")
        else:
            print(f" {BOLD}ArUco Estimation:{RESET} {RED}Waiting for data...{RESET}")
        print(f"==========================================================================")
        print(f" Press Ctrl+C to exit dashboard.")


def main(args=None):
    rclpy.init(args=args)
    node = CompareErrorsNode()
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
