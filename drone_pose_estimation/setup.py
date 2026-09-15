from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_pose_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your-github-username',
    maintainer_email='your_email@example.com',
    description='Drone Pose Estimation with ArUco and ICP',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_node = drone_pose_estimation.aruco_detector:main',
            'icp_node = drone_pose_estimation.icp_pose_estimator:main',
            'visualizer_node = drone_pose_estimation.visualize_icp:main',
            'evaluator_node = drone_pose_estimation.compare_errors:main',
            'icp_advanced_node = drone_pose_estimation.icp_pose_estimator_advanced:main',
            'icp_pure_node = drone_pose_estimation.icp_pose_estimator_pure:main',
            'kalman_fusion_node = drone_pose_estimation.kalman_filter_pose_fusion:main',
            'plotter_node = drone_pose_estimation.evaluation_plotter:main',
        ],
    },
)
