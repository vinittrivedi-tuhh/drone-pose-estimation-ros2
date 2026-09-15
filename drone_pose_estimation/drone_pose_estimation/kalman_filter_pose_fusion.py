#!/usr/bin/env python3
"""
================================================================================
 HYDROBUNNY — Kalman Filter Pose Fusion Node
 File: kalman_filter_pose_fusion.py
 ROS2 Jazzy | Python 3
================================================================================
"""

import time
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix
from drone_pose_estimation.common import CameraConfig

class KalmanFilterPoseFusion(Node):
    # Transformation parameters matching centralized calibration constants
    OFFSET_ARUCO_BASE_LINK = CameraConfig.OFFSET_ARUCO_BASE_LINK
    R_REF_MARKER_CONST = CameraConfig.R_REF_MARKER_CONST
    R_REF_DRONE = CameraConfig.R_REF_DRONE
    CAM_WORLD_POS = CameraConfig.CAM_WORLD_POS
    R_CAM_WORLD = CameraConfig.R_CAM_NOMINAL
    T_USD_TO_ROS = CameraConfig.T_USD_TO_ROS

    def __init__(self):
        super().__init__("kalman_filter_pose_fusion")

        # Configurable topic parameters
        self.declare_parameter("icp_topic", "/drone_pose_icp_advanced")
        self.declare_parameter("aruco_topic", "/drone_pose_aruco")

        icp_topic = self.get_parameter("icp_topic").get_parameter_value().string_value
        aruco_topic = self.get_parameter("aruco_topic").get_parameter_value().string_value

        # ── State Representation ──────────────────────────────────────────
        # x = [px, py, pz, vx, vy, vz, roll, pitch, yaw, w_roll, w_pitch, w_yaw]T
        # Length = 12
        self.state = np.zeros(12)
        
        # State Covariance matrix P (initialize with small uncertainty for position, larger for velocity)
        self.P = np.eye(12) * 0.1
        self.P[3:6, 3:6] *= 10.0      # Velocity uncertainty
        self.P[9:12, 9:12] *= 10.0    # Angular velocity uncertainty

        # Measurement matrix H (we measure 3D position and 3D Euler orientation)
        # Size: 6 x 12
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)   # Position measurement
        self.H[3:6, 6:9] = np.eye(3)   # Orientation measurement

        # ── Covariances ───────────────────────────────────────────────────
        # Process Noise Covariance Q (diagonal rates added per second)
        self.q_pos = 0.01              # Position drift (m/s)
        self.q_vel = 0.10              # Velocity change (m/s^2)
        self.q_rot = np.radians(1.0)   # Rotation drift (rad/s)
        self.q_ang_vel = np.radians(5.0) # Angular velocity change (rad/s^2)

        # Measurement Noise Covariance R for ArUco (Fast, higher noise)
        self.R_aruco = np.eye(6)
        self.R_aruco[0:3, 0:3] *= 0.03**2             # Translation variance (3 cm SD)
        self.R_aruco[3:6, 3:6] *= np.radians(4.0)**2  # Orientation variance (4 deg SD)

        # Measurement Noise Covariance R for ICP (Slow, extremely accurate)
        self.R_icp = np.eye(6)
        self.R_icp[0:3, 0:3] *= 0.006**2              # Translation variance (0.6 cm SD)
        self.R_icp[3:6, 3:6] *= np.radians(1.0)**2    # Orientation variance (1 deg SD)

        # ── Timing and Locks ──────────────────────────────────────────────
        self.lock = threading.Lock()
        self.last_update_time: float = self.get_clock().now().nanoseconds * 1e-9
        self.is_initialized = False

        # TF Buffer and Listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # QoS Setup
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscribers
        self.aruco_sub = self.create_subscription(
            PoseStamped, aruco_topic, self._aruco_callback, qos_profile=sensor_qos
        )
        self.icp_sub = self.create_subscription(
            PoseStamped, icp_topic, self._icp_callback, qos_profile=10
        )

        # Publishers
        self.fused_pub = self.create_publisher(
            PoseStamped, "/drone_pose_fused", qos_profile=10
        )

        # High-frequency state prediction / publication timer (30 Hz)
        self.timer_hz = 30.0
        self.timer = self.create_timer(1.0 / self.timer_hz, self._timer_callback)

        self.get_logger().info(f"[KalmanFilterPoseFusion] Node initialized. Running prediction at {self.timer_hz} Hz")

    def _predict(self, dt: float):
        """
        Run the state transition prediction step: x = F*x, P = F*P*F^T + Q
        """
        if dt <= 0.0:
            return

        # F matrix: constant-velocity model
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt  # p_new = p + v * dt
        F[6:9, 9:12] = np.eye(3) * dt # rot_new = rot + w * dt

        # Q matrix: process noise scaled by time
        Q = np.zeros((12, 12))
        Q[0:3, 0:3] = np.eye(3) * (self.q_pos**2 * dt)
        Q[3:6, 3:6] = np.eye(3) * (self.q_vel**2 * dt)
        Q[6:9, 6:9] = np.eye(3) * (self.q_rot**2 * dt)
        Q[9:12, 9:12] = np.eye(3) * (self.q_ang_vel**2 * dt)

        # Predict State
        self.state = F @ self.state
        # Angle wrap prediction state (Roll, Pitch, Yaw)
        self.state[6:9] = (self.state[6:9] + np.pi) % (2.0 * np.pi) - np.pi

        # Predict Covariance
        self.P = F @ self.P @ F.T + Q

    def _correct(self, z: np.ndarray, R: np.ndarray):
        """
        Run the Kalman correction step using a measurements vector z [x, y, z, roll, pitch, yaw]
        """
        # Measurement residual: y = z - H*x
        y = z - (self.H @ self.state)

        # CRITICAL: Normalize orientation residuals to [-pi, pi] to prevent angle wrapping jumps
        for i in range(3, 6):
            y[i] = (y[i] + np.pi) % (2.0 * np.pi) - np.pi

        # Innovation Covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + R

        # Kalman Gain: K = P*H^T*inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K*y
        self.state = self.state + K @ y
        # Re-normalize Roll, Pitch, Yaw
        self.state[6:9] = (self.state[6:9] + np.pi) % (2.0 * np.pi) - np.pi

        # Covariance update: Joseph form for numerical stability
        # P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I = np.eye(12)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def _aruco_callback(self, msg: PoseStamped):
        """
        ArUco callback: Shift ArUco to drone base_link, convert orientation to Euler, run KF correction.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        
        # 1. Transform raw ArUco pose to base_link position
        p_ar = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q_ar = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        R_ar = Rotation.from_quat(q_ar).as_matrix()
        
        t_base_link = p_ar + R_ar @ self.OFFSET_ARUCO_BASE_LINK
        
        # 2. Get orientation in Euler angles
        # Rotation mapping: ArUco marker frame → drone base_link frame
        # R_bl = R_aruco · R_REF_MARKER_CONST · R_REF_DRONE
        R_bl_ar = R_ar @ self.R_REF_MARKER_CONST @ self.R_REF_DRONE
        euler = Rotation.from_matrix(R_bl_ar).as_euler('xyz', degrees=False)  # radians

        z = np.hstack([t_base_link, euler])

        with self.lock:
            dt = now - self.last_update_time
            self.last_update_time = now

            if not self.is_initialized:
                self.state[0:3] = t_base_link
                self.state[6:9] = euler
                self.is_initialized = True
                self.get_logger().info("[KF Fusion] Initialized from ArUco measurement.")
                return

            self._predict(dt)
            self._correct(z, self.R_aruco)

    def _icp_callback(self, msg: PoseStamped):
        """
        ICP callback: ICP already represents base_link. Convert pose, run KF correction.
        """
        now = self.get_clock().now().nanoseconds * 1e-9

        t_base_link = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q_icp = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = Rotation.from_quat(q_icp).as_euler('xyz', degrees=False) # radians

        z = np.hstack([t_base_link, euler])

        with self.lock:
            dt = now - self.last_update_time
            self.last_update_time = now

            if not self.is_initialized:
                self.state[0:3] = t_base_link
                self.state[6:9] = euler
                self.is_initialized = True
                self.get_logger().info("[KF Fusion] Initialized from ICP measurement.")
                return

            self._predict(dt)
            self._correct(z, self.R_icp)

    def _timer_callback(self):
        """
        30 Hz prediction loop: Predict forward from last sensor event, publish fused pose, log GT error.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        
        with self.lock:
            if not self.is_initialized:
                return

            dt = now - self.last_update_time
            self.last_update_time = now
            self._predict(dt)
            
            # Extract position and orientation for publishing
            t_fused = np.copy(self.state[0:3])
            euler_fused = np.copy(self.state[6:9])

        # Publish pose
        r_fused = Rotation.from_euler('xyz', euler_fused, degrees=False)
        q_fused = r_fused.as_quat()

        fused_msg = PoseStamped()
        fused_msg.header.stamp = self.get_clock().now().to_msg()
        fused_msg.header.frame_id = "sim_camera"
        fused_msg.pose.position.x = float(t_fused[0])
        fused_msg.pose.position.y = float(t_fused[1])
        fused_msg.pose.position.z = float(t_fused[2])
        fused_msg.pose.orientation.x = float(q_fused[0])
        fused_msg.pose.orientation.y = float(q_fused[1])
        fused_msg.pose.orientation.z = float(q_fused[2])
        fused_msg.pose.orientation.w = float(q_fused[3])

        self.fused_pub.publish(fused_msg)

        # Log accuracy evaluation vs simulator GT
        self._evaluate_fused_pose(t_fused, r_fused.as_matrix())

    def _evaluate_fused_pose(self, t_fused, R_fused):
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
            T_u2r = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
            p_world_ros = T_u2r @ p_usd
            R_drone_ros_world = T_u2r @ Rotation.from_quat(q_usd).as_matrix()

            cam_pos = np.array([0.0, -1.22141, 2.35389])
            R_cam = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
            p_gt_ros = R_cam.T @ (p_world_ros - cam_pos)
            R_gt_ros = R_cam.T @ R_drone_ros_world

            # Calculate errors
            diff_t = t_fused - p_gt_ros
            err_3d = np.linalg.norm(diff_t)

            R_err = R_gt_ros.T @ R_fused
            tr_R_err = np.trace(R_err)
            val_clip = np.clip((tr_R_err - 1.0) / 2.0, -1.0, 1.0)
            err_rot_geo = np.degrees(np.arccos(val_clip))

            self.get_logger().info(
                f"[FUSED POSE] 3D Err: {err_3d*100:5.2f} cm | Geodesic Rot Err: {err_rot_geo:5.2f}°",
                throttle_duration_sec=1.0
            )

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    try:
        node = KalmanFilterPoseFusion()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
