# Copyright 2026 Li Wentian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
tennis_pick_launch.py
启动所有节点:
  1. tennis_vision_node     —— YOLO 视觉检测
  2. tennis_navigation_node —— SLAM 导航 / 底盘 / ToF 避障
  3. tennis_manipulation_node —— 机械臂 / 夹爪
  4. tennis_fsm_node        —— 状态机主控

用法:
  ros2 launch tennis_pick tennis_pick_launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch 参数
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='./best.pt',
        description='YOLO 模型路径'
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='是否无头模式运行 (无 GUI)'
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='',
        description='YOLO 推理设备: 空=自动, cpu, 0, 1...'
    )
    box_pose_arg = DeclareLaunchArgument(
        'box_pose',
        default_value='[]',
        description='放置点 [x, y, yaw_deg]，空列表则动态记录起点'
    )

    model_path = LaunchConfiguration('model_path')
    headless = LaunchConfiguration('headless')
    device = LaunchConfiguration('device')
    box_pose = LaunchConfiguration('box_pose')

    return LaunchDescription([
        model_path_arg,
        headless_arg,
        device_arg,
        box_pose_arg,

        # ---------- Vision Node ----------
        Node(
            package='tennis_pick',
            executable='tennis_vision_node',
            name='tennis_vision',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'camera_topic': '/camera/image_color',
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'imgsz': 640,
                'publish_debug_image': True,
                'device': device,
                'target_class': 'tennis',
            }],
        ),

        # ---------- Navigation Node ----------
        Node(
            package='tennis_pick',
            executable='tennis_navigation_node',
            name='tennis_navigation',
            output='screen',
            parameters=[{
                'tof_topic': '/range_0',
                'nav_pos_tolerance': 0.12,
                'nav_yaw_tolerance': 3.0,
                'nav_timeout': 60.0,
                'nav_forward_step': 0.3,
                'tof_trigger_m': 0.08,
                'chassis_linear_speed': 0.3,
                'chassis_angular_speed_deg': 45.0,
                'obstacle_stop_dist': 0.15,
                'map_frame': 'map',
                'base_frame': 'base_footprint',
            }],
        ),

        # ---------- Manipulation Node ----------
        Node(
            package='tennis_pick',
            executable='tennis_manipulation_node',
            name='tennis_manipulation',
            output='screen',
            parameters=[{
                'arm_home_x': 0.18,
                'arm_home_z': -0.07,
                'arm_grab_lift_x': 0.08,
                'arm_grab_lift_z': 0.11,
                'gripper_default_power': 0.7,
            }],
        ),

        # ---------- FSM Node ----------
        Node(
            package='tennis_pick',
            executable='tennis_fsm_node',
            name='tennis_fsm',
            output='screen',
            parameters=[{
                'headless': headless,
                'use_dynamic_box_pose': True,
                'box_pose': box_pose,
                'search_rotate_angle': 15.0,
                'grabbing_linear_speed': 0.18,
                'detect_confirm_frames': 3,
                'lost_trigger_frames': 5,
                'rotate_tolerance_px': 30,
            }],
        ),
    ])
