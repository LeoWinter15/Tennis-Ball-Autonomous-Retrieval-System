# tennis_pick

网球识别抓取 + SLAM 定点放置 (ROS2 Humble, Jetson + RoboMaster EP)

## 架构

参考 [BrickPick](https://github.com/KunLiangChen/BrickPick) 的模块化设计，将原单体节点拆分为 **4 个独立 ROS2 节点**，通过话题/服务松耦合通信：

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `tennis_vision_node` | YOLO 视觉检测 | `/camera/image_color` | `/tennis/detections`, `/tennis/annotated_image` |
| `tennis_navigation_node` | SLAM 位姿、底盘移动、ToF 避障 | `/range_0`, `/tennis/nav_goal`, `/tennis/cmd_vel_raw` | `/cmd_vel`, `/tennis/pose`, `/tennis/nav_status` |
| `tennis_manipulation_node` | 机械臂 + 夹爪控制 | Service: `/tennis/arm_move`, `/tennis/gripper_control`, `/tennis/arm_home` | Action: `/move_arm`, `/gripper` |
| `tennis_fsm_node` | 状态机主控 (SEARCH→ROTATE→GRAB→PLACE→HOME) | `/tennis/detections`, `/tennis/pose`, `/tennis/nav_status` | `/tennis/nav_goal`, `/tennis/cmd_vel_raw`, Service Clients |

## 状态机

```
SEARCHING ──发现球──> ROTATING ──对准──> GRABBING ──抓起──> PLACING ──放置──> HOMING ──归位──> SEARCHING
   ↑                                                                                          │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 依赖

- ROS2 Humble
- `robomaster_msgs` (robomaster_ros 包)
- `vision_msgs`
- YOLOv5 (`models/`, `utils/` 需放在 PYTHONPATH 或同目录)
- OpenCV, PyTorch, NumPy
- Jetson 上建议用 TensorRT 加速 YOLO

## 编译

```bash
cd ~/ros2_ws/src
cp -r <path_to_tennis_pick> .
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select tennis_pick
source install/setup.bash
```

## 运行

### 方式 1: Launch 一键启动

```bash
ros2 launch tennis_pick tennis_pick_launch.py
```

带参数：
```bash
ros2 launch tennis_pick tennis_pick_launch.py model_path:=/home/jetson/best.pt headless:=true
```

### 方式 2: 分别启动（调试用）

终端 1 —— 导航:
```bash
ros2 run tennis_pick tennis_navigation_node
```

终端 2 —— 机械臂:
```bash
ros2 run tennis_pick tennis_manipulation_node
```

终端 3 —— 视觉:
```bash
ros2 run tennis_pick tennis_vision_node --ros-args -p model_path:=./best.pt
```

终端 4 —— 状态机:
```bash
ros2 run tennis_pick tennis_fsm_node --ros-args -p headless:=true
```

## 话题与服务速查

### 话题

| 话题名 | 类型 | 说明 |
|--------|------|------|
| `/tennis/detections` | `vision_msgs/Detection2DArray` | 视觉检测结果 |
| `/tennis/annotated_image` | `sensor_msgs/Image` | 带标注的调试图像 |
| `/tennis/pose` | `geometry_msgs/PoseStamped` | 当前 SLAM 位姿 (map→base) |
| `/tennis/nav_status` | `std_msgs/String` | 导航状态: idle/navigating/success/failed/obstacle |
| `/tennis/nav_goal` | `geometry_msgs/PoseStamped` | 导航目标 (FSM → Navigation) |
| `/tennis/cmd_vel_raw` | `geometry_msgs/Twist` | FSM 原始速度指令 (Navigation 做安全/避障处理后下发) |

### 服务

| 服务名 | 类型 | 说明 |
|--------|------|------|
| `/tennis/arm_move` | `tennis_pick/srv/ArmMove` | 机械臂移动到指定 (x,z) |
| `/tennis/gripper_control` | `tennis_pick/srv/GripperControl` | 夹爪 open/close/pause |
| `/tennis/arm_home` | `std_srvs/Trigger` | 机械臂归位 |
| `/tennis/stop_nav` | `std_srvs/Trigger` | 停止当前导航 |

## 放置点 (Box Pose) 设置

### 方式 A: 动态记录（推荐）
启动时 FSM 节点自动读取当前 SLAM 位姿作为放置点：
```bash
ros2 run tennis_pick tennis_fsm_node --ros-args -p use_dynamic_box_pose:=true
```

### 方式 B: 手动指定
```bash
ros2 run tennis_pick tennis_fsm_node --ros-args -p box_pose:=[1.5,2.0,90.0]
```

## 避障说明

`tennis_navigation_node` 在收到 `/tennis/cmd_vel_raw` 时，会检查 ToF 距离：
- 若前方障碍物 < `obstacle_stop_dist` (默认 0.15m) 且正在前进，则禁止前进
- 导航过程中同样会检查 ToF，遇障则停止并返回 `obstacle` 状态

## 与原代码的主要改进

1. **模块化**: 4 个独立节点，可单独启停、复用、替换（如换用其他视觉模型只需改 vision_node）
2. **松耦合**: 节点间通过 ROS2 话题/服务通信，不共享内存
3. **避障集成**: Navigation 节点统一处理 ToF 避障，FSM 无需关心底层安全
4. **Service 抽象**: 机械臂/夹爪操作封装为 Service，FSM 通过简单调用即可
5. **导航异步**: FSM 发送 nav_goal 后异步等待 nav_status，不阻塞 ROS2 spin
6. **参数化**: 所有常量均可通过 ROS2 参数动态配置
7. **可扩展**: 新增传感器或行为只需添加新节点，不影响现有逻辑

## 文件结构

```
tennis_pick/
├── tennis_pick/
│   ├── __init__.py
│   ├── utils.py                  # 共享工具 (四元数转换等)
│   ├── tennis_vision_node.py     # 视觉节点
│   ├── tennis_navigation_node.py # 导航节点
│   ├── tennis_manipulation_node.py # 操作节点
│   └── tennis_fsm_node.py        # 状态机节点
├── srv/
│   ├── ArmMove.srv
│   └── GripperControl.srv
├── launch/
│   └── tennis_pick_launch.py
├── package.xml
├── setup.py
└── README.md
```

## License

Apache License 2.0
