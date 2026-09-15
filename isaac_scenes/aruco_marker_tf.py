import omni.usd
import numpy as np
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

# Get world transforms of both prims
xform_cache = UsdGeom.XformCache()

# Get ArUco marker world transform
aruco_prim = stage.GetPrimAtPath('/World/drone/base_link/aruco_marker')
aruco_world_xform = xform_cache.GetLocalToWorldTransform(aruco_prim)

# Get drone base_link world transform  
drone_prim = stage.GetPrimAtPath('/World/drone/base_link')
drone_world_xform = xform_cache.GetLocalToWorldTransform(drone_prim)

# T_marker_drone = inv(T_world_drone) * T_world_aruco
T_world_drone = np.array(drone_world_xform).reshape(4,4)
T_world_aruco = np.array(aruco_world_xform).reshape(4,4)

T_drone_aruco = np.linalg.inv(T_world_drone) @ T_world_aruco

print("=" * 50)
print("T_drone_aruco (4x4 matrix):")
print(np.round(T_drone_aruco, 6))

# Extract translation
tx = T_drone_aruco[0,3]
ty = T_drone_aruco[1,3]
tz = T_drone_aruco[2,3]
print(f"\nTranslation (x,y,z): [{tx:.6f}, {ty:.6f}, {tz:.6f}] meters")

# Extract rotation as euler angles
import scipy.spatial.transform as spt
R = T_drone_aruco[:3,:3]
euler = spt.Rotation.from_matrix(R).as_euler('xyz', degrees=True)
print(f"Rotation (roll,pitch,yaw): [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}] degrees")
print("=" * 50)
