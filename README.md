# 🎾 Tennis-Bot — Autonomous Tennis Ball Pick-and-Place Robot

> **Group 17** · Robotics & Embedded Systems Project · 2026

An autonomous mobile manipulator that **detects, navigates to, grasps and deposits tennis balls** on a court — fully unattended. Built on the DJI RoboMaster EP platform with a Jetson Orin NX edge computer, the system integrates YOLOv5 real-time detection, ROS 2 Nav2 autonomous navigation, SLAM mapping and Vosk Chinese voice control.

---

## Table of Contents

- [System Overview](#system-overview)
- [Hardware](#hardware)
- [Software Stack](#software-stack)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Phase 1 — Map Building](#phase-1--map-building-offline)
  - [Phase 2 — Autonomous Pick-and-Place](#phase-2--autonomous-pick-and-place)
- [State Machine](#state-machine)
- [Voice Control](#voice-control)
- [Key Parameters](#key-parameters)
- [Team](#team)
- [License](#license)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PERCEPTION LAYER                              │
│  USB Camera (640×360)  │  YDLidar X3  │  ToF IR  │  ReSpeaker 4-Mic│
└────────────┬───────────┴──────┬───────┴─────┬────┴────────┬────────┘
             │                  │             │             │
             ▼                  ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   COMPUTE & SOFTWARE LAYER                         │
│  Jetson Orin NX · ROS 2 Humble · YOLOv5 · Nav2 · SLAM · Vosk     │
└────────────┬───────────┬──────────────┬─────────────────────────────┘
             │           │              │
             ▼           ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ACTUATION LAYER                               │
│  Mecanum Chassis (/cmd_vel)  │  Robotic Arm  │  Gripper            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Component | Role | Notes |
|-----------|------|-------|
| **Jetson Orin NX** | Edge compute | ARM64 SoC with CUDA — runs all inference and ROS 2 nodes locally |
| **DJI RoboMaster EP** | Mobile platform | Mecanum-wheel chassis + built-in 2-DOF arm + parallel-jaw gripper |
| **YDLidar X3** | 2D LiDAR | 360° scan, 8 m range — used by SLAM Toolbox and AMCL |
| **USB RGB Camera** | Vision sensor | 640×360 @ 30 fps, feeds YOLOv5 detector |
| **ToF IR Sensor** | Proximity trigger | Mounted on EP front, triggers grasp at ≤ 0.13 m |
| **ReSpeaker 4-Mic Array** | Voice input | UAC 1.0, 16 kHz mono — feeds Vosk Chinese ASR |
| **Wooden Sensor Tower** | Custom mount | Elevates LiDAR, ReSpeaker and Jetson ~0.5 m above chassis center |

---

## Software Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Perception | YOLOv5 (PyTorch) | Real-time tennis ball detection |
| Vision Bridge | OpenCV + cv_bridge | Frame I/O, HUD overlay, debug visualization |
| Localization | SLAM Toolbox + AMCL | Build and localize against `tennis_field.pgm` |
| Navigation | Nav2 (NavigateToPose) | A* global planner + DWB local controller |
| Sensor Fusion | robot_localization (EKF) | Fuses wheel odometry → `/odom` |
| Voice I/O | Vosk (`small-cn-0.22`) | Offline Chinese keyword recognition — pause / resume |
| TF Tree | tf2_ros | Static `base_link → laser_frame` transform |
| Robot Driver | robomaster_ros | Chassis, arm, gripper, ToF action interfaces |

**ROS 2 Distribution:** Humble Hawksbill (Ubuntu 22.04)

---

## Repository Structure

```
yolov5-master/
├── tennis_pick_nav2_voice_mp.py   # Main control node (state machine + voice)
├── best.pt                        # YOLOv5 trained weights (tennis ball)
├── nav2_params.yaml               # Nav2 configuration (planner, DWB, costmap, AMCL)
├── ekf.yaml                       # robot_localization EKF config
├── vosk-model-small-cn-0.22/      # Vosk Chinese ASR model directory
├── models/                        # YOLOv5 model definitions
│   └── common.py
├── utils/                         # YOLOv5 utilities
└── ...

~/maps/
├── tennis_field.pgm               # Saved occupancy grid map
└── tennis_field.yaml              # Map metadata (resolution, origin)
```


### Phase 1 — Map Building (Offline)

Build a static occupancy grid map of the tennis court / indoor environment by teleoperating the robot.

```bash
# Terminal 1 — Start LiDAR driver
ros2 launch ydlidar_ros2_driver x3_ydlidar_launch.py

# Terminal 2 — Start DJI EP driver (with ToF sensor enabled)
ros2 launch robomaster_ros main.launch model:=ep conn_type:=sta tof_0:=true

# Terminal 3 — Publish static TF: base_link → laser_frame (LiDAR mounted 0.5m above base)
ros2 run tf2_ros static_transform_publisher 0 0 0.5 0 0 0 base_link laser_frame

# Terminal 4 — Start EKF node for odometry smoothing
ros2 run robot_localization ekf_node --ros-args --params-file ekf.yaml

# Terminal 5 — Start SLAM Toolbox (online async mode)
ros2 launch slam_toolbox online_async_launch.py

# Terminal 6 — Open RViz to visualize the map building process
ros2 run rviz2 rviz2
```

Now **teleoperate** the robot around the court to scan the environment:

```bash
# Terminal 7 — Keyboard teleoperation
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Once the map is complete (closed-loop scan of the court boundaries), **save the map**:

```bash
# Terminal 8 — Save the occupancy grid map
cd ~/maps
ros2 run nav2_map_server map_saver_cli -f tennis_field
```

This produces two files:
- `tennis_field.pgm` — the occupancy grid image (white = free, black = obstacle, gray = unknown)
- `tennis_field.yaml` — metadata (resolution, origin coordinates)

### Phase 2 — Autonomous Pick-and-Place

With the saved map, launch the full autonomous system:

```bash
# Terminal 1 — Start LiDAR driver
ros2 launch ydlidar_ros2_driver x3_ydlidar_launch.py

# Terminal 2 — Start DJI EP driver (with ToF sensor enabled)
ros2 launch robomaster_ros main.launch model:=ep conn_type:=sta tof_0:=true

# Terminal 3 — Publish static TF: base_link → laser_frame
ros2 run tf2_ros static_transform_publisher 0 0 0.5 0 0 0 base_link laser_frame

# Terminal 4 — Start Nav2 stack with the saved map
ros2 launch nav2_bringup bringup_launch.py \
    map:=/home/nvidia/maps/tennis_field.yaml \
    params_file:=/home/nvidia/Downloads/yolov5-17/yolov5-master/nav2_params.yaml \
    use_sim_time:=false

# Terminal 5 — Open RViz (set initial pose with "2D Pose Estimate" tool)
ros2 run rviz2 rviz2

# Terminal 6 — Run the main pick-and-place controller
cd ~/Downloads/yolov5-17/yolov5-master
python3 tennis_pick_nav2_voice_mp.py
```

> **Note:** After launching RViz, you must click **"2D Pose Estimate"** in the toolbar and set the robot's initial position on the map. Wait for the AMCL particle cloud to converge before the state machine starts.

---

## State Machine

The main control loop cycles through five states until no balls remain:

```
┌───────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐
│ SEARCHING │ ──► │ ROTATING │ ──► │ GRABBING │ ──► │ PLACING │ ──► │ HOMING │
└───────────┘     └──────────┘     └──────────┘     └─────────┘     └────────┘
      ▲                                                                  │
      └──────────────────── loop back ───────────────────────────────────┘
```

| State | Trigger to Enter | What It Does | Exit Condition |
|-------|-----------------|--------------|----------------|
| **SEARCHING** | Start / HOMING complete | Rotate slowly; run YOLOv5 on each frame | `detect_count ≥ 3` consecutive frames with confidence > 0.7 |
| **ROTATING** | Ball confirmed | Proportional yaw control to center ball in image | `\|pixel offset\| ≤ 30 px` (ROTATE_TOLERANCE_PX) |
| **GRABBING** | Ball centered | Drive forward + visual centering; open gripper | ToF distance ≤ 0.13 m → close gripper, lift arm |
| **PLACING** | Ball grasped | Call Nav2 `NavigateToPose` to bin location | Nav2 goal succeeded → open gripper, reverse, turn |
| **HOMING** | Ball deposited | Reset arm to home pose; clear counters | Immediate → back to SEARCHING |

### Async Inputs (run in parallel with all states)

- **Voice subprocess** — Vosk ASR in a `multiprocessing.Process`, sends `pause` / `resume` via `mp.Queue`
- **ToF Range topic** — triggers grasp in GRABBING state
- **AMCL pose** — provides localization for Nav2 planning

---

## Voice Control

A continuous voice monitoring layer runs in parallel with the state machine. The ReSpeaker 4-Mic Array feeds audio into an isolated Vosk subprocess.

| Command | Chinese Keywords | Action |
|---------|-----------------|--------|
| **Pause** | 停, 停下, 停止, 停车 | Immediately publishes zero `Twist` (×5), cancels active Nav2 goal, sets pause flag |
| **Resume** | 开始, 继续, 走, 出发 | Clears pause flag; state machine continues from where it stopped |

The pause/resume mechanism works across **all five states** — every motion routine calls `_wait_if_paused()` on entry, which blocks on a 100 ms idle loop until the flag is cleared.

---

## Key Parameters

All tunable constants are defined at the top of `tennis_pick_nav2_voice_mp.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_W × IMG_H` | 640 × 360 | Camera frame resolution |
| `ROTATE_TOLERANCE_PX` | 30 px | Max pixel offset to consider ball "centered" |
| `DETECT_CONFIRM_FRAMES` | 3 | Consecutive detection frames required to confirm a target |
| `LOST_TRIGGER_FRAMES` | 5 | Frames without detection before triggering a small-angle search |
| `TOF_TRIGGER_M` | 0.13 m | ToF distance threshold to trigger grasp |
| `CHASSIS_LINEAR_SPEED` | 0.3 m/s | Forward speed during GRABBING approach |
| `ARM_HOME_X, ARM_HOME_Z` | 0.18, -0.07 | Arm rest position (relative to base) |
| `ARM_GRAB_LIFT_X, ARM_GRAB_LIFT_Z` | 0.08, 0.11 | Arm lift position after grasping |
| `VOSK_MODEL_PATH` | `./vosk-model-small-cn-0.22` | Path to Vosk Chinese ASR model |

---

## Team

| Member | Role | Key Responsibilities |
|--------|------|---------------------|
| **Yanshan Wang** | System Design & Perception Lead | Overall framework design, YOLOv5 training & deployment, vision pipeline, ROS 2 environment setup |
| **Wenxin Xiao** | Mapping & Navigation Lead | SLAM environment setup, map building workflow, Nav2 stack configuration, PLACING state |
| **Wentian Li** | HMI & UI Lead | Operator console design, on-screen status overlay, demo recording & post-production |
| **Guanhong Chen** | Hardware & Voice Lead | Hardware bring-up & wiring, sensor calibration, Vosk voice subprocess, arm/gripper integration |

**Shared:** Map building sessions, on-court testing, final paper and presentation preparation.

---

