#!/usr/bin/env python3
"""
================================================================================
 HYDROBUNNY — ArUco Fiducial Marker Detector Node
 ROS 2 Jazzy | OpenCV | Python 3
================================================================================
"""

import os
import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf_transformations


class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        self.subscription = self.create_subscription(
            Image, "/rgb", self.image_callback, qos_profile_sensor_data
        )
        self.bridge = CvBridge()

        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, qos_profile_sensor_data
        )
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_size = 0.30

        # Robust ArUco Dictionary & Parameters
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        # Filters
        self.VALID_IDS = [0, 871]
        self.MAX_Z = 5.0
        self.MIN_Z = 0.3

        # Declare GUI visualization parameter
        self.declare_parameter("show_gui", True)
        self._show_gui = self.get_parameter("show_gui").get_parameter_value().bool_value

        self.pose_pub = self.create_publisher(PoseStamped, "/drone_pose_aruco", 10)
        self.frame_id = "sim_camera"

        self.get_logger().info(f"ArUco Detector started (OpenCV {cv2.__version__}) — filtering ID={self.VALID_IDS} Z=[{self.MIN_Z},{self.MAX_Z}]m")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None and self.camera_matrix is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id in enumerate(ids):
                mid = int(marker_id[0])
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]
                z = float(tvec[2])

                if mid not in self.VALID_IDS:
                    self.get_logger().warning(
                        f"Rejected ID {mid} (not in valid list {self.VALID_IDS})",
                        throttle_duration_sec=2.0
                    )
                    continue

                if z > self.MAX_Z or z < self.MIN_Z:
                    self.get_logger().warning(
                        f"Rejected ID {mid} — Z={z:.2f}m out of range [{self.MIN_Z},{self.MAX_Z}]m",
                        throttle_duration_sec=2.0
                    )
                    continue

                x_pos = float(tvec[0])
                y_pos = float(tvec[1])
                z_pos = float(tvec[2])

                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = tf_transformations.quaternion_from_matrix(
                    np.vstack([
                        np.hstack([R, np.array([[0], [0], [0]])]),
                        np.array([0, 0, 0, 1])
                    ])
                )

                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = self.frame_id
                pose_msg.pose.position.x = x_pos
                pose_msg.pose.position.y = y_pos
                pose_msg.pose.position.z = z_pos
                pose_msg.pose.orientation.x = float(qx)
                pose_msg.pose.orientation.y = float(qy)
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)

                self.pose_pub.publish(pose_msg)

                self.get_logger().info(
                    f"✅ ID {mid}: X:{x_pos:.3f} Y:{y_pos:.3f} Z:{z_pos:.3f}m"
                )

                cv2.drawFrameAxes(
                    cv_image, self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, self.marker_size * 0.5
                )
                aruco.drawDetectedMarkers(cv_image, [corners[i]], np.array([[mid]]))

        elif self.camera_matrix is None:
            self.get_logger().warning(
                "Waiting for camera intrinsics...", throttle_duration_sec=2.0
            )

        if self._show_gui and "DISPLAY" in os.environ:
            cv2.imshow("ArUco Detection Window", cv_image)
            cv2.waitKey(1)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received ✅")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
