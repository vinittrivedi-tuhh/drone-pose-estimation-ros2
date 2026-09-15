#!/usr/bin/env python3
"""
================================================================================
 HYDROBUNNY — ICP Pose Estimator Node
 File: icp_pose_estimator_node.py
 ROS2 Jazzy | Open3D | Python 3
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PIPELINE — PLAIN ENGLISH OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Goal: Give the UR30 arm a millimetre-accurate 6-DoF pose of the drone's
       fuel port so it can dock for hydrogen refuelling.

 Step 1 — Load the Digital Twin (done once at startup)
   We sample 10,000 points off the drone STL mesh.  This is our "reference"
   shape — the ideal, perfect drone in a coordinate system centered at the
   drone's body origin.  Think of it as a rubber stamp of what the drone
   looks like from above.

 Step 2 — ArUco gives us a coarse pose  (~cm accuracy, fast)
   The ArUco marker node already publishes: "the drone's ArUco marker is at
   roughly position X,Y,Z in camera frame, facing this direction."
   This is our starting guess — accurate enough to know roughly where to look,
   but not accurate enough for the arm.

 Step 3 — Convert coarse pose → model center in camera frame
   The ArUco marker is NOT at the drone's geometric center.  We measured the
   offset from live data: the drone body centroid is about [+0.509, -0.118,
   -0.529] m away from the marker in camera frame.  We apply this offset to
   get a predicted location for the drone body center in camera space.

 Step 4 — Crop & clean the live depth cloud
   The RealSense publishes millions of depth pixels as a 3D point cloud.
   We only keep points within a 2-metre box around our predicted drone center.
   This massively reduces the search space.  Then we voxel-downsample
   (merge nearby points into one) to ~5 cm resolution — faster ICP, same shape.

 Step 5 — Isolate the drone cluster (DBSCAN)
   After cropping, there may still be the platform, walls, or other clutter.
   DBSCAN is a clustering algorithm: it groups nearby points and ignores
   lone outlier points.  We pick the largest cluster — that should be the drone.

 Step 6 — ICP alignment  (the core refinement)
   ICP = Iterative Closest Point.  Imagine you have two jigsaw puzzle pieces
   that should fit together.  ICP repeatedly:
     (a) For each point in the reference, finds the nearest point in the live cloud
     (b) Computes a rigid transform (rotation + translation) that minimises the
         average distance between matched pairs
     (c) Applies that transform and repeats
   After ~50 iterations the reference shape "snaps" onto the live cloud.
   The final transform IS the drone pose in camera frame.

 Step 7 — Publish the refined pose
   We convert the 4×4 ICP transform matrix to a PoseStamped message and
   publish on /drone_pose_icp.  The arm controller reads this.

 Key quality metrics published alongside pose:
   • fitness  — fraction of reference points that found a close match (0→1)
                 We need ≥ 0.35 to trust the result.
   • inlier_rmse — average error of matched point pairs (metres)
                   We want < 0.08 m (8 cm) for usable accuracy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━════
"""
# Standrd libraries
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional,Tuple

# third-party libraries
import numpy as np
import open3d as o3d

# ROS2 libraries
from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# ROS2 message types
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# TF transformations
from tf_transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
)
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener
# SECTION 1: Configuration — all tunable values in one place
class Config:
    # STL Path
    # TODO: set this ROS parameter to your locally supplied confidential drone mesh.
    stl_path: str = ""
    # how many points to sample from the STL mesh for our reference cloud?
    # Rule of thumb: ~1 point per cm² of surface area is good for ICP.  The drone is about 1 m², so 10k points is a good balance of accuracy and speed.
    reference_n_points: int = 10_000

    # stl scale factor — if the STL is in mm, set to 0.001 to convert to metres
    stl_scale: float = 0.001

    # Cloud processing parameters
    crop_half_extent: float = 1.2   # metres around predicted mesh center


    #Voxel downsampling size — merge points within this distance to speed up ICP.  Too high = lose shape, too low = slow.
    voxel_size: float = 0.03

    # DBSCAN clustering parameters — to isolate the drone from background clutter
    # eps:      points within this distance (m) are considered neighbours
    # min_pts:  a core point needs at least this many neighbours
    # Keep eps ≈ 2× voxel_size so clusters aren't broken by downsampling gaps.
    dbscan_eps: float = 0.05
    dbscan_min_pts: int = 5

    #Minimum number of cluster points must have to be conssidered valid drone cluster
    min_cluster_points: int = 50

    # ICP parameters
    #Maximum distance (meters) for first icp pass
    #wide distance = tolerates large initial errors
    icp_max_dist_coarse: float = 0.5 #50 cm 
    icp_max_dist_fine: float = 0.08 # 8 cm

    # maximum number of ICP iterations
    icp_max_iterations: int = 400

    #Quality thresholds for accepting ICP result
    min_fitness: float = 0.20 # fraction of ref points matched
    max_rmse: float = 0.08    # preferred matched-pair average < 8 cm error
    max_rmse_loose: float = 0.20 # accept up to 20 cm if fitness is strong
    
    # ── Top-face filter ───────────────────────────────────────────────────────
    # -0.2 keeps everything except strongly downward-facing triangles.
    # top_face_normal_z_threshold: float = -0.2
    top_face_normal_z_threshold: float = 0
    # ── Z correction ──────────────────────────────────────────────────────────
    # height to get the body center, matching ArUco's coordinate convention.
    # Measure from your STL: after scale×0.001 + rotate90X the Z extent ≈ 0.56 m
    drone_half_height: float = 0.28   # metres — adjust if STL extent differs

    # Node timing
    pipeline_hz: float = 2.0  # run the whole pipeline at 2 Hz

# SECTION 2: STLLoader — Load STL, sample reference point cloud
class STLLoader:
    """
    Loads the drone STL file and turns it into a reference point cloud.

    WHY a point cloud instead of the mesh?
    ICP works on point clouds, not meshes.  We uniformly sample the mesh
    surface so every area is represented proportionally to its area.
    This is called Poisson-disk or uniform sampling.

    WHY keep only top-facing normals?
    The camera is mounted ABOVE the drone, so it can only see upward-facing
    surfaces.  Including bottom-facing surfaces in the reference would add
    points that can never appear in the live cloud, confusing ICP.
    We define "top-facing" as normals whose Z-component is positive after
    we centre the model (Z points upward in camera frame with camera looking
    down).  Adjust the sign if your camera orientation differs.
    """

    def __init__(self, cfg: Config, logger):
        self._cfg = cfg
        self._log = logger
        self.reference_pcd: Optional[o3d.geometry.PointCloud] = None
        self.drone_half_height: float = cfg.drone_half_height

    def load(self) -> bool:
        """
        Load STL, scale to metres, center at origin, filter visible faces,
        sample points, compute normals.  Returns True on success.
        """
        # Resolve path dynamically if it doesn't exist (makes it portable)
        path = self._cfg.stl_path
        if not path or not os.path.exists(path):
            self._log.error(
                "No local mesh is configured. The confidential CAD asset is not "
                "included; set the stl_path ROS parameter or DRONE_MESH_PATH."
            )
            return False

        self._log.info(f"[STLLoader] Loading STL: {path}")

        # Load mesh
        try:
            mesh = o3d.io.read_triangle_mesh(path)
        except Exception as e:
            self._log.error(f"[STLLoader] Failed to load STL: {e}")
            return False
        
        self._log.info(f"[STLLoader] Original mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

        # Scale to metres
        mesh.scale(self._cfg.stl_scale, center=(0,0,0))
        
        # ── ROTATE: STL height axis → camera depth axis ──────────────────────
        # WHY: The STL has the drone's thin axis (height=0.563m) along Y.
        # In camera frame, depth is Z and the camera looks DOWN (-Z world = +Z depth).
        # The drone's TOP surface (visible to camera) must map to SMALLER Z values
        # (closer to camera = smaller depth).
        # 
        # Rotation -90° around X:  Y → -Z  (height axis flips to depth, top faces -Z)
        # After T_init shifts by +1.07m in Z, top surface will be at Z ≈ 0.79m ✓
        #
        # NOTE: +90° was wrong — it put top surface at Z=+0.28 (far side),
        # making the reference upside-down relative to the live cloud.
        R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        mesh.rotate(R, center=np.zeros(3))
        
        # Center at origin
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        mesh.translate(-center) # now the drone's geometric center is at (0,0,0)

        self._log.info("[STLLoader] Mesh scaled, centered, and oriented to match camera frame")
        bbox2 = mesh.get_axis_aligned_bounding_box()
        ext = bbox2.get_extent()
        print(f"[STLLoader] After rotation extent: X={ext[0]:.2f}m  Y={ext[1]:.2f}m  Z={ext[2]:.2f}m")
        print("[STLLoader] Z should be ~0.56m (drone height facing camera)")
        
        # Compute triangle normals to update after scaling and rotation
        mesh.compute_triangle_normals()
        
        # -- Filter to top-facing triangles only -----------------------------
        # WHY: Camera looks DOWN at the drone. The visible surface faces UP in world
        # = faces the camera = has normals pointing toward -Z in reference frame
        # (after -90° X rotation, the drone's top surface normals point toward -Z).
        # We keep triangles with Z normal < +0.2 (i.e. NOT pointing away from camera).
        normals = np.asarray(mesh.triangle_normals)
        triangles = np.asarray(mesh.triangles)

        # Keep faces whose normal points TOWARD the camera (negative Z in ref frame)
        visible_mask      = normals[:, 2] < -self._cfg.top_face_normal_z_threshold
        visible_triangles = triangles[visible_mask]


        if len(visible_triangles) == 0:
            self._log.warn(
                "[STLLoader] No top-facing triangles found!  "
                "Using full mesh.  Check camera orientation."
            )
        else:
            mesh.triangles = o3d.utility.Vector3iVector(visible_triangles)
            self._log.info(
                f"[STLLoader] Kept {visible_mask.sum():,}/{len(normals):,} "
                "top-facing triangles"
            )
        
        # -- Sample points uniformly from the mesh surface -------------------
        # WHY Poisson-disk sampling: uniform surface sampling ensures even
        # coverage — no over-representation of small dense-triangle regions.
        self.reference_pcd = mesh.sample_points_poisson_disk(
            number_of_points=self._cfg.reference_n_points
        )

        # -- Estimate normals on the sampled point cloud ---------------------
        # WHY: Open3D's point-to-plane ICP variant uses surface normals to
        # get faster, more accurate convergence than point-to-point ICP.
        # We estimate them from the local neighbourhood (KNN with k=20).
        self.reference_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        # Orient normals consistently upward (toward +Z, i.e., toward camera)
        self.reference_pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 1.0])
        )

        n_pts = len(self.reference_pcd.points)
        self._log.info(f"[STLLoader] Reference PCD ready: {n_pts:,} points")

        # ── DIAGNOSTIC: print reference PCD extent so we can verify orientation ──
        pts = np.asarray(self.reference_pcd.points)
        print(f"[STLLoader] Ref PCD X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
        print(f"[STLLoader] Ref PCD Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
        print(f"[STLLoader] Ref PCD Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
        print(f"[STLLoader] Ref PCD centroid: {pts.mean(axis=0).round(4)}")
        print(f"[STLLoader] Ref PCD sample[0]: {pts[0].round(4)}")
        print(f"[STLLoader] Ref PCD sample[100]: {pts[100].round(4)}")
        return True

#Section 3 CLoudProcessor — crop, downsample, cluster live cloud
class CloudProcessor:
    """
    Processes the raw live point cloud from the RealSense depth camera:
      1. Convert ROS PointCloud2 → Open3D PointCloud
      2. Crop to a bounding box around the predicted drone center
      3. Voxel-downsample (reduce density while preserving shape)
      4. DBSCAN clustering — isolate the largest cluster (the drone)

    WHY crop first?
    The full depth cloud is 640×480 = ~307 200 points.  Doing ICP on all of
    them would be ~10× slower than needed.  Cropping to a 3 m box around the
    predicted center reduces points to ~5 000–30 000.

    WHY voxel-downsample?
    ICP is O(N log N) in the number of points.  Reducing from 30 000 → 3 000
    points (10× fewer) gives ~100× speedup with negligible accuracy loss,
    because the drone shape is preserved at 3 cm resolution.

    WHY DBSCAN?
    Even after cropping, the cloud contains the platform the drone sits on,
    the walls, and possibly parts of the arm.  DBSCAN separates spatially
    distinct clusters without knowing in advance how many there are.  We take
    the largest cluster — statistically the drone body.
    """
    
    def __init__(self, cfg: Config, logger):
        self._cfg = cfg
        self._log = logger
    
    def ros_to_open3d(self, msg: PointCloud2) -> o3d.geometry.PointCloud:
        """
        Convert ROS PointCloud2 message to Open3D PointCloud.
        """
        # Read points from ROS PointCloud2 using high-performance numpy buffer reading
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)

        if pts.shape[0] == 0:
            self._log.warn("[CloudProcessor] Received empty PointCloud2!")
            return o3d.geometry.PointCloud()

        # Convert from Isaac Sim coordinates (meters) to Open3D PCD

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pcd
    
    def crop(self, pcd: o3d.geometry.PointCloud, center: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Crop the point cloud to a cube of side 2×crop_half_extent around the center.
        """
        h = self._cfg.crop_half_extent
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=center - h, max_bound=center + h)
        cropped = pcd.crop(bbox)
        self._log.debug(
            f"[CloudProcessor] Crop: {len(pcd.points):,} → "
            f"{len(cropped.points):,} points  (center {center})"
        )
        return cropped
    
    def voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Replace the cloud with one point per voxel_size³ voxel.

        HOW it works: the 3D space is divided into a regular grid of cubes
        (voxels).  All points inside each cube are averaged into one point.
        This is lossless in terms of shape fidelity at scales larger than
        the voxel size.
        """
        voxel_size = self._cfg.voxel_size
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        self._log.debug(
            f"[CloudProcessor] Voxel-downsample: {len(pcd.points):,} → "
            f"{len(downsampled.points):,} points  (voxel size {voxel_size})"
        )
        return downsampled
    
    def isolate_drone_cluster(self, pcd: o3d.geometry.PointCloud) -> Tuple[Optional[o3d.geometry.PointCloud], int]:
        """
        Use DBSCAN clustering to isolate the drone from background clutter.
        Returns the largest cluster, which should be the drone.

        DBSCAN parameters:
          eps:      points within this distance (m) are considered neighbours
          min_pts:  a core point needs at least this many neighbours
          Core points + their reachable neighbours = one cluster.
          Points not reachable from any core point = noise (label = -1).

        WHY we want the LARGEST cluster:
          The drone body occupies the most volume and therefore the most points
          in the cropped region.  Walls, floor patches, and arm segments are
          either sparse noise or much smaller clusters.

        Returns: (cluster_pcd, n_points) or (None, 0) on failure.
        """
        if len(pcd.points) == 0:
            self._log.warn("[CloudProcessor] No points to cluster!")
            return None, 0
        
        labels = np.array(pcd.cluster_dbscan(
            eps=self._cfg.dbscan_eps, min_points=self._cfg.dbscan_min_pts,
            print_progress=False))
        
        #labels == -1 are noise points; ignore them
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            self._log.warn("[CloudProcessor] DBSCAN found no clusters!")
            return None, 0

        # Find the largest cluster by number of points
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        best_label = max(cluster_sizes, key=cluster_sizes.get)
        best_size = cluster_sizes[best_label]

        self._log.info(
            f"[CloudProcessor] DBSCAN found {len(unique_labels)} clusters; "
            f"largest is label {best_label} with {best_size} points"
            f"All sizes: { {k: v for k,v in sorted(cluster_sizes.items(), key=lambda x: -x[1])[:5]} }"
        )

        if best_size < self._cfg.min_cluster_points:
            self._log.warn(
                f"[CloudProcessor] Largest cluster has only {best_size} points, "
                f"which is below the threshold of {self._cfg.min_cluster_points}. "
                "Check crop size and ArUco Z accuracy."
            )
            return None, 0
        #Build a point cloud of the largest cluster
        mask = labels == best_label
        cluster_pts = np.asarray(pcd.points)[mask]
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)

        return cluster_pcd, best_size
    
    def process(self, ros_msg: PointCloud2, predicted_center : np.ndarray) -> Tuple[Optional[o3d.geometry.PointCloud], int]:
        """
        Full processing pipeline: ROS → Open3D, crop, downsample, cluster.
        Returns the processed point cloud of the drone cluster, or None on failure.
        """
        pcd = self.ros_to_open3d(ros_msg)
        if len(pcd.points) == 0:
            return None, 0

        pcd = self.crop(pcd, predicted_center)
        if len(pcd.points) == 0:
            return None, 0

        pcd = self.voxel_downsample(pcd)

        return self.isolate_drone_cluster(pcd)

#Section 4 ICPRunner - Align reference PCD to live cluster

class ICPRunner:
    """
    Runs ICP alignment of the reference PCD onto the live cluster.

    Two-pass strategy (coarse → fine):
      Pass 1 — large correspondence distance (0.5 m)
        Allows big positional errors to be corrected first.
        The reference might be off by 20–30 cm from the initial guess.
        Starting with a tight threshold here would find NO correspondences
        and immediately fail.

      Pass 2 — tight correspondence distance (0.05 m)
        Refinement pass.  Now that the reference is roughly in place,
        we tighten the matching to get millimetre-level accuracy.

    ICP variant: Point-to-Plane
      Standard ICP minimises the SUM OF SQUARED DISTANCES between matched
      point pairs (point-to-point).  Point-to-plane ICP instead minimises
      the distance FROM each source point TO THE TANGENT PLANE of the matched
      target point.  This converges ~6× faster on smooth surfaces because it
      correctly handles sliding along flat regions.  We precomputed normals
      on the reference during STLLoader.load().

    Convergence criteria:
      ICP stops when either:
        (a) max_iterations is reached, OR
        (b) the transform change between iterations is < relative_fitness
            and < relative_rmse (we use Open3D defaults for these).

    Result interpretation:
      fitness:     fraction of reference points that found a match within
                   the correspondence distance.  1.0 = perfect match.
                   Our camera only sees the top of the drone, so 0.35–0.60
                   is realistic and acceptable.
      inlier_rmse: mean distance between matched pairs.  Think of this as
                   the average "miss distance."  < 5 cm = good for our task.
    """

    def __init__(self, cfg: Config, logger):
        self._cfg = cfg
        self._log = logger

    def run(
        self,
        reference_pcd: o3d.geometry.PointCloud,
        live_cluster:  o3d.geometry.PointCloud,
        T_init:        np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Align reference_pcd (the digital twin) onto live_cluster using ICP.

        Args:
            reference_pcd  : reference point cloud, centered at origin.
            live_cluster   : live drone cluster from depth camera.
            T_init         : 4×4 initial transform placing reference near cluster.

        Returns:
            (T_final, fitness, rmse)
            T_final is None if ICP quality is below thresholds.
        """
        live_cluster.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        # Camera is at Z=0 relative to itself; drone is at Z>0 (positive depth).
        # Orient normals toward the camera = toward smaller Z = toward [0, 0, 0].
        live_cluster.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

        # -- Convergence criteria (Open3D standard) --------------------------
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self._cfg.icp_max_iterations,
        )
        # -- Pass 1: Coarse ICP (large correspondence distance) --------------
        # WHY: Our initial guess from ArUco may still be 20–40 cm off.
        # A 50 cm max-distance lets ICP "reach out" and find correspondences
        # even when the reference is not yet well-aligned.
        self._log.info(
            f"[ICPRunner] Pass 1 (coarse): max_dist={self._cfg.icp_max_dist_coarse} m"
        )
        t0 = time.monotonic()
        result_coarse = o3d.pipelines.registration.registration_icp(
            source=reference_pcd,             # what we align
            target=live_cluster,              # what we align TO
            max_correspondence_distance=self._cfg.icp_max_dist_coarse,
            init=T_init,                      # initial transform (ArUco guess)
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ),
            criteria=criteria,
        )
        t1 = time.monotonic()
        self._log.info(
            f"[ICPRunner] Pass 1 done in {(t1-t0)*1000:.1f} ms  "
            f"fitness={result_coarse.fitness:.3f}  "
            f"rmse={result_coarse.inlier_rmse*100:.1f} cm"
        )

        # -- Pass 2: Fine ICP (tight correspondence distance) ----------------
        # We start from Pass 1's result as the new initial guess.
        # WHY two passes? After Pass 1 the reference is roughly aligned.
        # Tightening the distance now only matches very close pairs,
        # pulling the pose to millimetre accuracy without being misled by
        # distant (possibly incorrect) associations from the coarse pass.
        fine_max_dist = self._cfg.icp_max_dist_fine
        self._log.info(
            f"[ICPRunner] Pass 2 (fine):   max_dist={fine_max_dist:.3f} m"
        )
        t0 = time.monotonic()
        result_fine = o3d.pipelines.registration.registration_icp(
            source=reference_pcd,
            target=live_cluster,
            max_correspondence_distance=fine_max_dist,
            init=result_coarse.transformation,   # use Pass 1 output
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ),
            criteria=criteria,
        )
        t1 = time.monotonic()
        self._log.info(
            f"[ICPRunner] Pass 2 done in {(t1-t0)*1000:.1f} ms  "
            f"fitness={result_fine.fitness:.3f}  "
            f"rmse={result_fine.inlier_rmse*100:.1f} cm"
        )

        if result_fine.fitness == 0.0:
            self._log.warn(
                "[ICPRunner] Pass 2 point-to-point failed with zero fitness; "
                "attempting point-to-plane fallback."
            )
            t0 = time.monotonic()
            result_fine = o3d.pipelines.registration.registration_icp(
                source=reference_pcd,
                target=live_cluster,
                max_correspondence_distance=fine_max_dist,
                init=result_coarse.transformation,
                estimation_method=(
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                ),
                criteria=criteria,
            )
            t1 = time.monotonic()
            self._log.info(
                f"[ICPRunner] Fallback Pass 2 done in {(t1-t0)*1000:.1f} ms  "
                f"fitness={result_fine.fitness:.3f}  "
                f"rmse={result_fine.inlier_rmse*100:.1f} cm"
            )

        fitness = result_fine.fitness
        rmse    = result_fine.inlier_rmse
        T_final = np.array(result_fine.transformation)   # 4×4 matrix

        # -- Quality gate 
        if fitness < self._cfg.min_fitness:
            self._log.warn(
                f"[ICPRunner] fitness={fitness:.3f} < threshold {self._cfg.min_fitness} — "
                "rejecting result.  Possible causes: cluster too sparse, "
                "wrong crop center, or reference PCD orientation mismatch."
            )
            return None, fitness, rmse

        if rmse > self._cfg.max_rmse:
            if fitness >= 0.75 and rmse <= self._cfg.max_rmse_loose:
                self._log.warn(
                    f"[ICPRunner] rmse={rmse*100:.1f} cm exceeds strict "
                    f"threshold {self._cfg.max_rmse*100:.0f} cm, "
                    f"but fitness={fitness:.3f} is strong; accepting result."
                )
            else:
                self._log.warn(
                    f"[ICPRunner] rmse={rmse*100:.1f} cm > threshold "
                    f"{self._cfg.max_rmse*100:.0f} cm — rejecting result."
                )
                return None, fitness, rmse

        self._log.info(
            f"[ICPRunner] ✓ ICP accepted: fitness={fitness:.3f}, "
            f"rmse={rmse*100:.1f} cm"
        )
        return T_final, fitness, rmse

#Section 5: ICPEstimatorNode — ROS2 node tying everything together
class ICPEstimatorNode(Node):
        """
    ROS2 node that runs the full ICP pose estimation pipeline at 2 Hz.

    Subscriptions:
      /drone_pose_aruco  (geometry_msgs/PoseStamped) — coarse ArUco pose
      /depth_pcl         (sensor_msgs/PointCloud2)   — live depth cloud

    Publications:
      /drone_pose_icp    (geometry_msgs/PoseStamped) — refined 6-DoF pose

    Threading model:
      Both subscribers write their latest messages to instance variables
      protected by a lock.  The timer callback (running in the ROS executor
      thread) reads the latest values, runs the pipeline, and publishes.
      This avoids blocking the subscriber callbacks and handles cases where
      PCL and ArUco arrive at different rates.
    """
        def __init__(self):
            super().__init__("icp_pose_estimator")

            # Configuration 
            self._cfg = Config()

            # Build sub-modules ─
            self._stl_loader    = STLLoader(self._cfg, self.get_logger())
            self._cloud_proc    = CloudProcessor(self._cfg, self.get_logger())
            self._icp_runner    = ICPRunner(self._cfg, self.get_logger())

            # Load reference PCD once at startup
            if not self._stl_loader.load():
                self.get_logger().fatal(
                    "[ICPEstimatorNode] Failed to load STL reference PCD.  "
                    "Node will not function.  Check stl_path in Config."
                )
                raise RuntimeError("STL load failed")
            self._half_height = self._stl_loader.drone_half_height

            # TF buffer and listener for ground truth pose lookups
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            # Shared state (subscriber → pipeline) 
            self._lock        = threading.Lock()
            self._aruco_msg:  Optional[PoseStamped]   = None
            self._pcl_msg:    Optional[PointCloud2]   = None
            self._last_aruco: float = 0.0     # timestamp of last ArUco msg
            self._last_pcl:   float = 0.0     # timestamp of last PCL msg
            self._msg_timeout = 2.0           # seconds before we consider stale

            # QoS profile 
            # Best-effort + volatile = matches Isaac Sim's default sensor topics.
            # Change to RELIABLE if your setup uses it.
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            # ── Subscribers ───────────────────────────────────────────────────
            self._aruco_sub = self.create_subscription(
                PoseStamped,
                "/drone_pose_aruco",
                self._aruco_callback,
                qos_profile=sensor_qos,
            )
            self._pcl_sub = self.create_subscription(
                PointCloud2,
                "/depth_pcl",
                self._pcl_callback,
                qos_profile=sensor_qos,
            )

            # ── Publisher ─────────────────────────────────────────────────────
            self._pose_pub = self.create_publisher(
                PoseStamped,
                "/drone_pose_icp",
                qos_profile=10,
            )

            # ── Pipeline timer (2 Hz) ─────────────────────────────────────────
            period = 1.0 / self._cfg.pipeline_hz
            self._timer = self.create_timer(period, self._pipeline_callback)

            self.get_logger().info(
                "[ICPEstimatorNode] ✓ Node initialised.  "
                f"Running pipeline at {self._cfg.pipeline_hz} Hz."
            )

        # ── Subscriber callbacks ──────────────────────────────────────────────────

        def _aruco_callback(self, msg: PoseStamped) -> None:
            """Cache the latest ArUco pose.  Non-blocking."""
            with self._lock:
                self._aruco_msg  = msg
                self._last_aruco = self.get_clock().now().nanoseconds * 1e-9

        def _pcl_callback(self, msg: PointCloud2) -> None:
            """Cache the latest depth point cloud.  Non-blocking."""
            with self._lock:
                self._pcl_msg  = msg
                self._last_pcl = self.get_clock().now().nanoseconds * 1e-9

        # ── Pipeline callback (timer) ─────────────────────────────────────────────
        def _pipeline_callback(self) -> None:
            now = self.get_clock().now().nanoseconds * 1e-9

            with self._lock:
                if (now - self._last_aruco) > self._msg_timeout if self._last_aruco else True:
                    self.get_logger().warn("[ICPEstimatorNode] ArUco pose stale")
                    return
                if (now - self._last_pcl) > self._msg_timeout if self._last_pcl else True:
                    self.get_logger().warn("[ICPEstimatorNode] Depth PCL stale")
                    return
                aruco_msg = self._aruco_msg
                pcl_msg   = self._pcl_msg

            if aruco_msg is None or pcl_msg is None:
                self.get_logger().warn("Waiting for messages…"); return

            # ── Step 1: ArUco marker position and rotation in ROS camera frame ────
            aruco_pos = np.array([
                aruco_msg.pose.position.x,
                aruco_msg.pose.position.y,
                aruco_msg.pose.position.z,
            ])
            q_aruco = [
                aruco_msg.pose.orientation.x,
                aruco_msg.pose.orientation.y,
                aruco_msg.pose.orientation.z,
                aruco_msg.pose.orientation.w,
            ]
            R_aruco = Rotation.from_quat(q_aruco).as_matrix()

            # ── Step 2: Compute predicted drone mesh center in camera frame ───────
            # Constant offset from marker to drone mesh center in OpenCV ArUco frame:
            offset_mesh_aruco = np.array([-0.24767, 0.01116, -0.00497])
            predicted_mesh_center = aruco_pos + R_aruco @ offset_mesh_aruco

            self.get_logger().info(
                f"[ICPEstimatorNode] ArUco (ROS): ({aruco_pos[0]:.3f},{aruco_pos[1]:.3f},{aruco_pos[2]:.3f})  "
                f"→ Predicted Mesh Center: ({predicted_mesh_center[0]:.3f},{predicted_mesh_center[1]:.3f},{predicted_mesh_center[2]:.3f})"
            )

            # ── Step 3: Crop around predicted mesh center in camera frame ─────────
            live_cluster, n_pts = self._cloud_proc.process(pcl_msg, predicted_mesh_center)
            if live_cluster is None or n_pts == 0:
                self.get_logger().warn("[ICPEstimatorNode] No drone cluster — skipping")
                return

            # ── Step 5: Build T_init in camera frame ──────────────────────────────
            # Constant rotation mapping ArUco orientation to reference mesh:
            R_ref_marker_const = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
            R_init = R_aruco @ R_ref_marker_const
            
            T_init = np.eye(4)
            T_init[:3, :3] = R_init
            T_init[:3, 3] = predicted_mesh_center
            
            yaw_init = Rotation.from_matrix(R_init).as_euler('xyz', degrees=True)[2]
            self.get_logger().info(
                f"[ICPEstimatorNode] T_init: pos=({T_init[0,3]:.3f},{T_init[1,3]:.3f},{T_init[2,3]:.3f})  yaw={yaw_init:.1f}°"
            )

            # ── Step 6: Run ICP ───────────────────────────────────────────────────
            T_final, fitness, rmse = self._icp_runner.run(
                reference_pcd=self._stl_loader.reference_pcd,
                live_cluster=live_cluster,
                T_init=T_init,
            )

            if T_final is None:
                self.get_logger().warn(
                    f"[ICPEstimatorNode] ICP rejected (fitness={fitness:.3f}, rmse={rmse*100:.1f} cm)"
                )
                return

            # ── Step 7: Convert mesh center to base_link pose ──────────────────────
            # Rotation from drone base_link to reference mesh frame:
            R_ref_drone = np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0],
                [0.0, -1.0,  0.0],
            ])
            R_final = np.array(T_final[:3, :3])
            R_ros_drone = R_final @ R_ref_drone
            
            # Translate estimated mesh center back to base_link origin:
            t_base_link = T_final[:3, 3] + R_ros_drone @ np.array([-0.00700, -0.14600, -0.64400])
            
            # Build T_base_link matrix for publishing
            T_base_link = np.eye(4)
            T_base_link[:3, :3] = R_ros_drone
            T_base_link[:3, 3] = t_base_link

            # ── Step 9: Publish ───────────────────────────────────────────────────
            pose_msg = self._transform_to_pose_stamped(T_base_link, aruco_msg.header.frame_id)
            self._pose_pub.publish(pose_msg)

            # ── Step 10: Log vs Ground Truth ──────────────────────────────────────
            # Look up true base_link pose from /tf directly in camera frame
            try:
                trans = self._tf_buffer.lookup_transform("sim_camera", "drone", rclpy.time.Time())
                
                # Since sim_camera is now oriented directly in the ROS RDF convention,
                # we can read the Ground Truth relative translation and orientation directly from TF.
                p_gt_ros = np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ])

                q_gt_ros = np.array([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])
                R_gt_ros = Rotation.from_quat(q_gt_ros).as_matrix()
                
                err_icp = np.linalg.norm(t_base_link - p_gt_ros)
                
                # Predict ArUco base_link position for comparison
                aruco_base_link = aruco_pos + R_aruco @ np.array([-0.89167, 0.00416, -0.15097])
                err_aruco = np.linalg.norm(aruco_base_link - p_gt_ros)
                
                self.get_logger().info(
                    f"[ICPEstimatorNode] ✓ Comparison vs Ground Truth:\n"
                    f"  ICP Base_Link ROS:   pos=({t_base_link[0]:.3f},{t_base_link[1]:.3f},{t_base_link[2]:.3f})  err={err_icp*100:.2f} cm\n"
                    f"  ArUco Base_Link ROS: pos=({aruco_base_link[0]:.3f},{aruco_base_link[1]:.3f},{aruco_base_link[2]:.3f})  err={err_aruco*100:.2f} cm\n"
                    f"  GT Base_Link ROS:    pos=({p_gt_ros[0]:.3f},{p_gt_ros[1]:.3f},{p_gt_ros[2]:.3f})"
                )
            except Exception as e:
                self.get_logger().warn(f"GT Comparison failed: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────
        def _transform_to_pose_stamped(self, T: np.ndarray, frame_id: str) -> PoseStamped:
                msg                  = PoseStamped()
                msg.header.stamp     = self.get_clock().now().to_msg()
                msg.header.frame_id  = frame_id
                msg.pose.position.x  = float(T[0, 3])
                msg.pose.position.y  = float(T[1, 3])
                msg.pose.position.z  = float(T[2, 3])
                q = quaternion_from_matrix(T)
                msg.pose.orientation.x = float(q[0])
                msg.pose.orientation.y = float(q[1])
                msg.pose.orientation.z = float(q[2])
                msg.pose.orientation.w = float(q[3])
                return msg
 
 

# SECTION 7: Entry point


def main(args=None):
    """Standard ROS2 Python node entry point."""
    rclpy.init(args=args)

    try:
        node = ICPEstimatorNode()
    except RuntimeError as exc:
        print(f"[FATAL] Node init failed: {exc}")
        rclpy.shutdown()
        return

    try:
        # spin() hands control to the ROS executor which manages callbacks
        # and timers until the node is killed (Ctrl-C / ros2 lifecycle stop).
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
