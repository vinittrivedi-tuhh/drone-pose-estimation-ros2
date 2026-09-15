#!/usr/bin/env python3
"""
================================================================================
 HYDROBUNNY — Automated Pose Evaluation Logger & Plotter (Comparative Version)
 File: evaluation_plotter.py
 ROS2 Jazzy | Python 3
================================================================================
"""

import os
import csv
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from drone_pose_estimation.common import CameraConfig, resolve_results_directory

class EvaluationPlotter(Node):
    OFFSET_ARUCO_BASE_LINK = CameraConfig.OFFSET_ARUCO_BASE_LINK
    R_REF_MARKER_CONST = CameraConfig.R_REF_MARKER_CONST
    R_REF_DRONE = CameraConfig.R_REF_DRONE
    CAM_WORLD_POS = CameraConfig.CAM_WORLD_POS
    R_CAM_WORLD = CameraConfig.R_CAM_NOMINAL
    T_USD_TO_ROS = CameraConfig.T_USD_TO_ROS

    def __init__(self):
        super().__init__("evaluation_plotter")

        # TF Buffer and Listener to get simulator Ground Truth
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_publish_static_camera()

        # Threading lock
        self._lock = threading.Lock()

        # Data cache list
        self._data_log = []
        self._start_time = time.time()

        # Declare ROS 2 parameters
        self.declare_parameter("test_name", "isaac_sim_nominal")
        self.declare_parameter("duration_sec", 30.0)
        self.declare_parameter("cam_pitch_deg", -90.0)
        self.declare_parameter("cam_roll_deg", 0.0)
        self.declare_parameter("cam_yaw_deg", 180.0)

        test = self.get_parameter("test_name").get_parameter_value().string_value
        if not test:
            test = "isaac_sim_nominal"
        self._duration_sec = self.get_parameter("duration_sec").get_parameter_value().double_value
        self._cam_pitch = self.get_parameter("cam_pitch_deg").get_parameter_value().double_value
        self._cam_roll = self.get_parameter("cam_roll_deg").get_parameter_value().double_value
        self._cam_yaw = self.get_parameter("cam_yaw_deg").get_parameter_value().double_value

        # Auto-parse camera angles from standard test name pattern (e.g. x_-80_y_-5_z_170)
        import re
        m = re.search(r'x_([-\d]+)_y_([-\d]+)_z_([-\d]+)', test)
        if m:
            self._cam_pitch = float(m.group(1))
            self._cam_roll = float(m.group(2))
            self._cam_yaw = float(m.group(3))
            self.get_logger().info(f"[EvaluationPlotter] Auto-detected camera angles from test_name: pitch={self._cam_pitch}°, roll={self._cam_roll}°, yaw={self._cam_yaw}°")

        # Output Directories dynamically resolved without hardcoded usernames
        csv_dir = resolve_results_directory("csv")
        png_dir = resolve_results_directory("png")
        
        self._csv_path = os.path.join(csv_dir, f"evaluation_results_{test}.csv")
        self._png_path = os.path.join(png_dir, f"evaluation_plots_{test}.png")

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Cache variables for latest callbacks & wall-time of arrival
        self._latest_aruco = None
        self._aruco_recv_time = 0.0

        self._latest_icp_base = None
        self._icp_base_recv_time = 0.0

        self._latest_icp_adv = None
        self._icp_adv_recv_time = 0.0

        self._latest_icp_pure = None
        self._icp_pure_recv_time = 0.0

        self._latest_fused = None
        self._fused_recv_time = 0.0

        # Subscriptions
        self._aruco_sub = self.create_subscription(
            PoseStamped, "/drone_pose_aruco", self._aruco_callback, qos_profile=sensor_qos
        )
        self._icp_adv_sub = self.create_subscription(
            PoseStamped, "/drone_pose_icp_advanced", self._icp_adv_callback, qos_profile=10
        )
        self._icp_pure_sub = self.create_subscription(
            PoseStamped, "/drone_pose_icp_pure", self._icp_pure_callback, qos_profile=10
        )

        # Timer to sample and compute errors at 10 Hz
        self._sample_rate = 10.0
        self._timer = self.create_timer(1.0 / self._sample_rate, self._sample_callback)

        self.get_logger().info(f"Evaluation Plotter Node started. Saving results to: {self._csv_path}")

    def _aruco_callback(self, msg: PoseStamped):
        with self._lock:
            self._latest_aruco = msg
            self._aruco_recv_time = time.time()

    def _icp_base_callback(self, msg: PoseStamped):
        with self._lock:
            self._latest_icp_base = msg
            self._icp_base_recv_time = time.time()

    def _icp_adv_callback(self, msg: PoseStamped):
        with self._lock:
            self._latest_icp_adv = msg
            self._icp_adv_recv_time = time.time()

    def _icp_pure_callback(self, msg: PoseStamped):
        with self._lock:
            self._latest_icp_pure = msg
            self._icp_pure_recv_time = time.time()

    def _fused_callback(self, msg: PoseStamped):
        with self._lock:
            self._latest_fused = msg
            self._fused_recv_time = time.time()

    def _get_gt_pose(self) -> tuple:
        """
        Query ground truth pose of the drone in ROS camera frame.
        """
        try:
            # Query Dynamic Camera Pose from TF
            try:
                cam_trans = self._tf_buffer.lookup_transform("world", "sim_camera", rclpy.time.Time())
                cam_world_pos = np.array([
                    cam_trans.transform.translation.x,
                    cam_trans.transform.translation.y,
                    cam_trans.transform.translation.z
                ])
                q_cam_world = np.array([
                    cam_trans.transform.rotation.x,
                    cam_trans.transform.rotation.y,
                    cam_trans.transform.rotation.z,
                    cam_trans.transform.rotation.w
                ])
                R_cam_world = Rotation.from_quat(q_cam_world).as_matrix()
            except Exception:
                # Fallback — Isaac Sim: Translate(-0.3,-1.3,0.0), Orient(90°X,90°Y,0°Z)
                # USD(Y-up) → ROS(Z-up): x=-0.3, y=0.0, z=-1.3
                cam_world_pos = np.array([0.0, -1.22141, 2.35389])
                # Rx(90°)·Ry(90°) rotation matrix
                R_cam_world = np.array([
                    [ 0.0,  0.0,  1.0],
                    [ 1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0],
                ])

            # Isaac Sim publishes 'drone' in USD Y-up world coords; camera TF is ROS Z-up.
            # Fix: look up world→drone, convert USD→ROS, project to camera frame manually.
            trans_gt = self._tf_buffer.lookup_transform("world", "drone", rclpy.time.Time())

            p_usd = np.array([
                trans_gt.transform.translation.x,
                trans_gt.transform.translation.y,
                trans_gt.transform.translation.z
            ])
            q_usd = np.array([
                trans_gt.transform.rotation.x,
                trans_gt.transform.rotation.y,
                trans_gt.transform.rotation.z,
                trans_gt.transform.rotation.w
            ])

            # USD Y-up → ROS Z-up: x_ros=x_usd, y_ros=-z_usd, z_ros=y_usd
            T_u2r = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
            p_world_ros = T_u2r @ p_usd
            R_drone_usd = Rotation.from_quat(q_usd).as_matrix()
            R_drone_ros_world = T_u2r @ R_drone_usd

            # Camera pose in ROS Z-up world
            cam_pos = np.array([0.0, -1.22141, 2.35389])
            R_cam_nom = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])

            # Apply camera tilt perturbation if test is non-nominal
            delta_pitch = self._cam_pitch - (-90.0)
            delta_roll = self._cam_roll - 0.0
            delta_yaw = self._cam_yaw - 180.0
            if abs(delta_pitch) > 1e-3 or abs(delta_roll) > 1e-3 or abs(delta_yaw) > 1e-3:
                r_delta = Rotation.from_euler('xyz', [delta_pitch, delta_roll, delta_yaw], degrees=True)
                R_cam = R_cam_nom @ r_delta.as_matrix()
            else:
                R_cam = R_cam_nom

            p_gt_ros = R_cam.T @ (p_world_ros - cam_pos)
            R_gt_ros = R_cam.T @ R_drone_ros_world

            return p_gt_ros, R_gt_ros

        except Exception as e:
            return None, None

    def _sample_callback(self):
        t_now = time.time() - self._start_time
        if t_now >= self._duration_sec:
            self.get_logger().info(f"⏰ {self._duration_sec} seconds limit reached! Shutting down evaluation_plotter and generating plots...")
            raise SystemExit

        # 1. Fetch GT Pose
        p_gt_ros, R_gt_ros = self._get_gt_pose()
        if p_gt_ros is None or R_gt_ros is None:
            # TF transform not available yet
            return

        gt_euler = Rotation.from_matrix(R_gt_ros).as_euler('xyz', degrees=True)

        with self._lock:
            aruco_msg = self._latest_aruco
            aruco_recv = self._aruco_recv_time

            icp_base_msg = self._latest_icp_base
            icp_base_recv = self._icp_base_recv_time

            icp_adv_msg = self._latest_icp_adv
            icp_adv_recv = self._icp_adv_recv_time

            icp_pure_msg = self._latest_icp_pure
            icp_pure_recv = self._icp_pure_recv_time

            fused_msg = self._latest_fused
            fused_recv = self._fused_recv_time

        now_system = time.time()

        # Helper function to compute translation and orientation error
        def process_msg(msg, recv_time, is_aruco=False):
            if msg is None:
                return [np.nan]*3 + [np.nan]*3 + [np.nan, np.nan]
            
            # Check if message is older than 1.5 seconds in wall-time (stale)
            if (now_system - recv_time) > 1.5:
                return [np.nan]*3 + [np.nan]*3 + [np.nan, np.nan]
            
            p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            q = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            R_est = Rotation.from_quat(q).as_matrix()
            
            if is_aruco:
                # Shift raw ArUco to drone base_link
                pos_bl = p + R_est @ self.OFFSET_ARUCO_BASE_LINK
                R_bl = R_est @ self.R_REF_MARKER_CONST @ self.R_REF_DRONE
            else:
                pos_bl = p
                R_bl = R_est
            
            euler_est = Rotation.from_matrix(R_bl).as_euler('xyz', degrees=True)
            
            # Compute errors
            err_3d = np.linalg.norm(pos_bl - p_gt_ros) * 100.0  # in cm
            
            R_err = R_gt_ros.T @ R_bl
            tr_R_err = np.trace(R_err)
            val_clip = np.clip((tr_R_err - 1.0) / 2.0, -1.0, 1.0)
            err_rot = np.degrees(np.arccos(val_clip))
            
            return [pos_bl[0], pos_bl[1], pos_bl[2], euler_est[0], euler_est[1], euler_est[2], err_3d, err_rot]

        # Process all streams
        aruco_data = process_msg(aruco_msg, aruco_recv, is_aruco=True)
        icp_base_data = process_msg(icp_base_msg, icp_base_recv, is_aruco=False)
        icp_adv_data = process_msg(icp_adv_msg, icp_adv_recv, is_aruco=False)
        icp_pure_data = process_msg(icp_pure_msg, icp_pure_recv, is_aruco=False)

        # Save to log list
        self._data_log.append([t_now, p_gt_ros[0], p_gt_ros[1], p_gt_ros[2], gt_euler[0], gt_euler[1], gt_euler[2]] 
                              + aruco_data + icp_base_data + icp_adv_data + icp_pure_data)

    def save_and_plot(self):
        """
        Called when node is destroyed or shutdown. Saves log to CSV and generates matplotlib plots.
        """
        if not self._data_log:
            print("[EvaluationPlotter] No data logged. Skipping CSV save and plot generation.")
            return

        print(f"[EvaluationPlotter] Saving {len(self._data_log)} records to CSV...")
        
        headers = [
            "timestamp", "gt_x", "gt_y", "gt_z", "gt_roll", "gt_pitch", "gt_yaw",
            "aruco_x", "aruco_y", "aruco_z", "aruco_roll", "aruco_pitch", "aruco_yaw", "aruco_err_3d", "aruco_err_rot",
            "icp_base_x", "icp_base_y", "icp_base_z", "icp_base_roll", "icp_base_pitch", "icp_base_yaw", "icp_base_err_3d", "icp_base_err_rot",
            "icp_adv_x", "icp_adv_y", "icp_adv_z", "icp_adv_roll", "icp_adv_pitch", "icp_adv_yaw", "icp_adv_err_3d", "icp_adv_err_rot",
            "icp_pure_x", "icp_pure_y", "icp_pure_z", "icp_pure_roll", "icp_pure_pitch", "icp_pure_yaw", "icp_pure_err_3d", "icp_pure_err_rot"
        ]

        try:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self._data_log)
            print(f"[EvaluationPlotter] ✅ CSV successfully saved to: {self._csv_path}")
        except Exception as e:
            print(f"[EvaluationPlotter] Failed to write CSV: {str(e)}")

        print("[EvaluationPlotter] Generating comparative plots...")
        try:
            import matplotlib.pyplot as plt
            
            data = np.array(self._data_log)
            t = data[:, 0]
            
            # Errors extraction
            aruco_err_3d  = data[:, 13]
            aruco_err_rot = data[:, 14]
            icp_base_err_3d  = data[:, 21]
            icp_base_err_rot = data[:, 22]
            icp_adv_err_3d   = data[:, 29]
            icp_adv_err_rot  = data[:, 30]
            icp_pure_err_3d  = data[:, 37]
            icp_pure_err_rot = data[:, 38]

            fig, axs = plt.subplots(2, 2, figsize=(16, 11))
            fig.suptitle("HYDROBUNNY — Comparative Pose Estimation Analysis", fontsize=18, fontweight='bold')

            # Styles for the 3 primary estimation methods
            styles = [
                (aruco_err_3d, aruco_err_rot, "ArUco (with Offset)", "orange", 1.2),
                (icp_adv_err_3d, icp_adv_err_rot, "Advanced ICP (Tracking)", "cyan", 1.4),
                (icp_pure_err_3d, icp_pure_err_rot, "Pure ICP (Markerless)", "magenta", 1.2),
            ]

            # Plot 1: 3D Translation Error over Time
            for err_3d, _, label, color, lw in styles:
                axs[0, 0].plot(t, err_3d, label=label, color=color, linewidth=lw, alpha=0.8)
            axs[0, 0].set_title("Translational 3D Euclidean Error", fontweight='bold', fontsize=12)
            axs[0, 0].set_xlabel("Time (s)")
            axs[0, 0].set_ylabel("Error (cm)")
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            # Plot 2: Rotational Geodesic Error over Time
            for _, err_rot, label, color, lw in styles:
                axs[0, 1].plot(t, err_rot, label=label, color=color, linewidth=lw, alpha=0.8)
            axs[0, 1].set_title("Rotational Geodesic Error", fontweight='bold', fontsize=12)
            axs[0, 1].set_xlabel("Time (s)")
            axs[0, 1].set_ylabel("Error (deg)")
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            # Helper for CDF calculation
            def plot_cdf(ax, err_data, label, color):
                valid_data = err_data[~np.isnan(err_data)]
                if len(valid_data) == 0:
                    return
                sorted_data = np.sort(valid_data)
                y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, y_cdf * 100.0, label=label, color=color, linewidth=2)

            # Plot 3: CDF of Translation Error
            for err_3d, _, label, color, _ in styles:
                plot_cdf(axs[1, 0], err_3d, label, color)
            axs[1, 0].set_title("Translation Error CDF", fontweight='bold', fontsize=12)
            axs[1, 0].set_xlabel("Error (cm)")
            axs[1, 0].set_ylabel("Percentage of Frames (%)")
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            # Plot 4: CDF of Rotation Error
            for _, err_rot, label, color, _ in styles:
                plot_cdf(axs[1, 1], err_rot, label, color)
            axs[1, 1].set_title("Rotation Error CDF", fontweight='bold', fontsize=12)
            axs[1, 1].set_xlabel("Error (deg)")
            axs[1, 1].set_ylabel("Percentage of Frames (%)")
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            plt.tight_layout()
            plt.savefig(self._png_path)
            plt.close()
            print(f"[EvaluationPlotter] ✅ Comparative plots successfully saved to: {self._png_path}")

        except ImportError:
            print("[EvaluationPlotter] matplotlib package not found. Skipping plot image generation. (CSV is still saved!)")
        except Exception as e:
            print(f"[EvaluationPlotter] Failed to generate plots: {str(e)}")

    def _tf_publish_static_camera(self):
        """Publish static TF: world → sim_camera.

        Camera settings from Isaac Sim (Camera_OmniVision_OV9782_Color):
          - Translate : X=-0.3, Y=-1.3, Z=0.0  (Isaac Sim USD Y-up frame)
          - Orient    : X=90°, Y=90°, Z=0°
          - Focal Length : 1.5 mm | Aperture (H) : 3.896 mm
          - fx=fy≈492.81 px @ 1280×720 | Horizontal FoV ≈ 104.7°

        USD(Y-up) → ROS-world(Z-up): x=-0.3, y=0.0, z=-1.3
        Rx(90°)·Ry(90°) → q = (0.5, 0.5, -0.5, 0.5)
        """
        self._static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'sim_camera'
        t.transform.translation.x = 0.0
        t.transform.translation.y = -1.22141
        t.transform.translation.z = 2.35389
        # Rx(90°)·Ry(90°) → q = (0.5, 0.5, -0.5, 0.5)
        t.transform.rotation.x =  0.5
        t.transform.rotation.y =  0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w =  0.5
        self._static_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_plot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
