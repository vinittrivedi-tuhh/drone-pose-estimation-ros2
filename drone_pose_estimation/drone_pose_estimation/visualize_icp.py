#!/usr/bin/env python3
"""
HYDROBUNNY — ICP Live Pose 3D Visualizer App
ROS2 node and Open3D GUI app to visualize the registered drone model in real-time.
"""

import os
import threading
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

# Enforce DISPLAY settings for X11 compatibility
os.environ.setdefault("DISPLAY", ":0")
os.environ["XDG_SESSION_TYPE"] = "x11"

class ICPVisualizerApp(Node):
    """
    Object-Oriented Open3D Visualizer Application.
    Manages the ROS2 subscription to estimated poses and updates the 
    3D digital twin visualization in real-time.
    """
    
    def __init__(self, stl_path: str, topic: str):
        super().__init__("icp_visualizer")
        self._stl_path = stl_path
        self._topic = topic

        # Cache variables for thread-safe pose updates
        self._latest_T = None
        self._lock = threading.Lock()

        # Load reference digital twin mesh
        self._ref_pcd = self._load_reference_pcd()

        # Open3D Visualizer core object
        self._vis = o3d.visualization.Visualizer()

        # Subscriptions
        self._sub = self.create_subscription(
            PoseStamped, 
            self._topic, 
            self._pose_callback, 
            10
        )
        self.get_logger().info(f"Subscribed to {self._topic}")

    def _pose_callback(self, msg: PoseStamped):
        """Callback to cache the incoming estimated pose stamps (thread-safe)."""
        p = msg.pose.position
        q = msg.pose.orientation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        
        # Build 4x4 homogenous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = p.x
        T[1, 3] = p.y
        T[2, 3] = p.z
        
        with self._lock:
            self._latest_T = T

    def _load_reference_pcd(self) -> o3d.geometry.PointCloud:
        """Loads, scales, centers, and samples the CAD reference STL model."""
        self.get_logger().info(f"Loading reference mesh: {self._stl_path}")
        mesh = o3d.io.read_triangle_mesh(self._stl_path)
        
        # Scale to meters (from mm)
        mesh.scale(0.001, center=np.zeros(3))
        
        # Rotate from STL height axis to camera optical depth axis
        R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        mesh.rotate(R, center=np.zeros(3))
        
        # Center mesh at its geometric origin
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-bbox.get_center())
        
        # Sample points uniformly for the point cloud representation
        pcd = mesh.sample_points_uniformly(10000)
        pcd.paint_uniform_color([1, 0, 0]) # Red color for visualization
        return pcd

    def run(self):
        """Creates the GUI window and starts the main rendering event loop."""
        self._vis.create_window("ICP Live Pose", width=1200, height=800)
        opt = self._vis.get_render_option()

        # Handle OpenGL/GLX context errors gracefully
        if opt is None:
            print("\n\033[91m[ERROR] Open3D failed to create OpenGL context or window.\033[0m")
            print("This usually happens in VMs, Docker, or headless SSH sessions.")
            print("Please try running with software rendering enabled:")
            print("\033[1m  export LIBGL_ALWAYS_SOFTWARE=1\033[0m")
            print("\033[1m  export __GLX_VENDOR_LIBRARY_NAME=mesa\033[0m")
            print("\033[1m  python3 visualize_icp.py\033[0m\n")
            import os
            os._exit(1)

        # Initialize visualizer geometries
        display_pcd = o3d.geometry.PointCloud()
        display_pcd.points = o3d.utility.Vector3dVector(np.asarray(self._ref_pcd.points))
        display_pcd.colors = o3d.utility.Vector3dVector(np.asarray(self._ref_pcd.colors))
        self._vis.add_geometry(display_pcd)

        # Add coordinate frame at the camera origin
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self._vis.add_geometry(axis)

        # Style options
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 2.0
        opt.light_on = False  # flat coloring

        # Spin ROS in background thread to handle subscription callbacks
        ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        ros_thread.start()

        prev_T = None
        print(f"Waiting for pose on '{self._topic}' — Rotate: mouse drag | Zoom: scroll | Quit: Q")

        # Main rendering loop
        while self._vis.poll_events():
            with self._lock:
                T = self._latest_T

            # Apply estimated transform dynamically if it updates
            if T is not None and (prev_T is None or not np.allclose(T, prev_T)):
                if prev_T is not None:
                    display_pcd.transform(np.linalg.inv(prev_T)) # undo previous
                display_pcd.transform(T) # apply new
                self._vis.update_geometry(display_pcd)
                prev_T = T.copy()

            self._vis.update_renderer()

        # Clean up
        self._vis.destroy_window()


def main(args=None):
    rclpy.init(args=args)
    
    # The confidential CAD mesh is not distributed with this repository.
    STL_PATH = os.environ.get("DRONE_MESH_PATH", "")
    if not STL_PATH:
        raise RuntimeError("Set DRONE_MESH_PATH to a locally supplied drone mesh before visualizing ICP.")

    TOPIC = "/drone_pose_icp"
    app = ICPVisualizerApp(STL_PATH, TOPIC)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
