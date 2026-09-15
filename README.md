# HYDRO-BUNNY — Drone Pose Estimation for Autonomous Hydrogen Refuelling

A ROS 2 pipeline for real-time 6-DoF pose estimation of a fixed-wing UAV, built to support autonomous robotic hydrogen refuelling using a UR30 industrial manipulator. Developed as my Projektarbeit at the Institute of Aircraft Production Technology (IFPT), Hamburg University of Technology (TUHH) — part of the HYDRO-BUNNY project.

**Grade:** 1.3

> ⚠️ **What's not in this repo:** All CAD/mesh assets (`.stl`, `.usd`) for the drone, landing platform, and rail system are the property of MB+Partner and are not included here, per project confidentiality. This repo contains only the perception pipeline, evaluation tooling, and results that are my own work.

---

## The problem

Refuelling a hydrogen-powered drone autonomously — with zero human contact, for hours at a time — requires a robot arm that can find the drone's exact position and orientation on its own. This repo is the perception system that makes that possible: three independent 6-DoF pose estimators, benchmarked against each other under realistic tilt and lighting conditions in NVIDIA Isaac Sim.

---

## Three estimators, one comparison

| Node | Approach | Marker dependency |
| :--- | :--- | :--- |
| `aruco_node` | Fiducial marker detection (OpenCV `cv2.aruco`, sub-pixel corner refinement, PnP solve) | Required, every frame |
| `icp_advanced_node` | ArUco-seeded initialization, then temporal point-cloud tracking (Point-to-Plane ICP from the previous frame's pose) | Required once, at start only |
| `icp_pure_node` | Fully markerless — mounting structure volume carving, 8-candidate multi-start yaw search (45° steps), landing-pad-prior symmetry disambiguation | None |

A 12-state Kalman filter (`kalman_fusion_node`, 30 Hz) is implemented to fuse ArUco and ICP outputs — see [Honest limitations](#honest-limitations) for its actual status.

---

## Key results

All results from NVIDIA Isaac Sim, 10 Hz logging, 6 drone orientations × 3 lighting conditions. Translation error is 3D Euclidean distance to ground truth; rotation error is geodesic angle.

### Nominal condition (straight-down, flat drone)

| Method | Mean 3D error | Mean rotation error |
| :--- | :--- | :--- |
| **ArUco** | **0.22 cm** | **0.30°** |
| **Advanced ICP** | 0.91 cm | 0.70° |
| **Pure ICP** | 0.69 cm | 0.55° |

*Under ideal conditions, ArUco wins — reported honestly, not hidden.*

### Combined tilt stress (x=-80°, y=-5°, z=170°)

| Method | Mean 3D error |
| :--- | :--- |
| **ArUco** | 15.54 cm — ~70× its nominal error |
| **Advanced ICP** | 1.44 cm |
| **Pure ICP** | **0.62 cm** |

### Illumination sweep — the clearest result in the project

| Condition | ArUco | Advanced ICP | Pure ICP |
| :--- | :--- | :--- | :--- |
| **Low light** | 1.56 cm | 0.50 cm | 0.52 cm |
| **Nominal light** | 2.95 cm | 0.48 cm | 0.49 cm |
| **Bright glare** | 3.00 cm | 0.50 cm | 0.52 cm |

ArUco's error increases under normal/bright lighting (visible-light glare confuses marker detection); both ICP methods stay flat regardless of lighting, since active infrared depth sensing doesn't depend on ambient light.

*Footnote:* the pure-yaw stress test (x=-90°, y=0°, z=170°) is reported elsewhere at 290 frames, but the retained raw log for that run only contains 86 rows (~8.5 s). Treat that specific configuration's numbers as a shorter sample than the others.

---

## Honest limitations

This project reports what was actually measured — nothing more:

- **Simulation-only.** All results are from Isaac Sim; no physical hardware validation has been performed yet.
- **Stationary drone only.** Every test uses one static pose per configuration — no landing dynamics, approach velocity, or vibration was evaluated.
- **Kalman fusion is implemented, not evaluated.** A topic-subscription configuration issue meant the fusion node didn't receive live ICP input during recorded runs, so no quantitative fusion results exist. It's in the code as future work, not a result.
- **Pure ICP relies on a known position prior, not unconstrained global search** — it assumes the drone is within a known docking-area region.
- **Zero CAD-to-object mismatch, by construction** — the reference mesh and the simulated target come from the same asset, so simulation accuracy doesn't include the manufacturing-tolerance error a physical system would have.
- **Sub-millimetre docking precision is not yet achieved.** Nominal translation errors were 2.2 mm (ArUco), 9.1 mm (Advanced ICP), 6.9 mm (Pure ICP) — good enough for coarse robot positioning, not yet for final fuel-coupling docking.

---

## Tech stack

ROS 2 (Jazzy / Humble) · NVIDIA Isaac Sim · Open3D (ICP, DBSCAN, Poisson sampling) · OpenCV (`cv2.aruco`) · NumPy / SciPy · Matplotlib · Pandas

---

## Repository structure

```text
drone_pose_estimation/
├── aruco_detector.py              # ArUco fiducial pose estimation
├── icp_pose_estimator.py          # Baseline ICP (2-pass, DBSCAN-cropped)
├── icp_pose_estimator_advanced.py # ArUco-seeded temporal ICP tracking
├── icp_pose_estimator_pure.py     # Fully markerless ICP (multi-start yaw)
├── kalman_filter_pose_fusion.py   # 12-state Kalman fusion (implemented, unevaluated)
├── evaluation_plotter.py          # CSV + plot generation for all benchmarks
├── compare_errors.py              # Live terminal error dashboard
├── visualize_icp.py               # Open3D real-time registration viewer
├── common.py                      # Shared preprocessing (voxel/carving/filters)
└── calibration_tools/             # Offline calibration utilities
results/                           # Raw CSV logs + generated comparison plots
```

---

## Future work

- Physical hardware validation (real RealSense D435 + real drone)
- Evaluate the Kalman fusion filter after fixing the topic-subscription issue
- Closed-loop UR30 integration (MoveIt 2) and an actual hydrogen coupling trial
- Second, end-effector-mounted camera for the final sub-millimetre approach
- Dynamic/moving-drone tracking with IMU fusion

---

## Acknowledgements

Supervised by Prof. Dr.-Ing. Thorsten Schüppstuhl and M.Sc. Omar Draidrya, Institute of Aircraft Production Technology (IFPT), TUHH. Part of the HYDRO-BUNNY project. #hydrobunny
