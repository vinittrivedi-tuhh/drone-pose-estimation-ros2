import re
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    test_name = LaunchConfiguration("test_name").perform(context)
    duration_sec = LaunchConfiguration("duration_sec").perform(context)

    # Auto-parse camera tilt angles from test_name (e.g., isaac_sim_x_-80_y_-5_z_170)
    cam_pitch = -90.0
    cam_roll = 0.0
    cam_yaw = 180.0

    m = re.search(r'x_([-\d]+)_y_([-\d]+)_z_([-\d]+)', test_name)
    if m:
        cam_pitch = float(m.group(1))
        cam_roll = float(m.group(2))
        cam_yaw = float(m.group(3))
        print(f"\n[Launch] ✅ Detected camera tilt: Pitch={cam_pitch}°, Roll={cam_roll}°, Yaw={cam_yaw}°\n")
    else:
        print(f"\n[Launch] ℹ️ Using nominal camera orientation: Pitch={cam_pitch}°, Roll={cam_roll}°, Yaw={cam_yaw}°\n")

    params = {
        "cam_pitch_deg": cam_pitch,
        "cam_roll_deg": cam_roll,
        "cam_yaw_deg": cam_yaw,
    }

    icp_advanced_node = Node(
        package="drone_pose_estimation",
        executable="icp_advanced_node",
        name="advanced_icp_pose_estimator",
        parameters=[params],
        output="screen",
    )

    plotter_node = Node(
        package="drone_pose_estimation",
        executable="plotter_node",
        name="evaluation_plotter",
        parameters=[{
            "test_name": test_name,
            "duration_sec": float(duration_sec),
            **params
        }],
        output="screen",
    )

    return [icp_advanced_node, plotter_node]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "test_name",
            default_value="isaac_sim_nominal",
            description="Test name (e.g. isaac_sim_x_-80_y_-5_z_170 or isaac_sim_nominal)"
        ),
        DeclareLaunchArgument(
            "duration_sec",
            default_value="30.0",
            description="Duration in seconds for plotter recording"
        ),
        OpaqueFunction(function=launch_setup)
    ])
