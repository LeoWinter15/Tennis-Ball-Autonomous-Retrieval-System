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
tennis_manipulation_node.py
机械臂操作节点 —— 负责机械臂和夹爪控制

服务:
  /tennis/arm_move        (tennis_pick/srv/ArmMove)
  /tennis/gripper_control (tennis_pick/srv/GripperControl)
  /tennis/arm_home        (std_srvs/Trigger)

内部通过 Action Client 调用:
  /move_arm  (robomaster_msgs/action/MoveArm)
  /gripper   (robomaster_msgs/action/GripperControl)
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import Trigger
from tennis_pick.srv import ArmMove, GripperControl
from robomaster_msgs.action import MoveArm, GripperControl as GripperAction

import time

# 机械臂常量 (米，相对 arm_base_link)
ARM_HOME_X, ARM_HOME_Z = 0.18, -0.07
ARM_GRAB_LIFT_X, ARM_GRAB_LIFT_Z = 0.08, 0.11

# GripperControl 状态常量
GRIPPER_PAUSE = 0
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 2


class TennisManipulationNode(Node):
    def __init__(self):
        super().__init__('tennis_manipulation')

        # ---------- 参数 ----------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('arm_home_x', ARM_HOME_X),
                ('arm_home_z', ARM_HOME_Z),
                ('arm_grab_lift_x', ARM_GRAB_LIFT_X),
                ('arm_grab_lift_z', ARM_GRAB_LIFT_Z),
                ('gripper_default_power', 0.7),
            ]
        )
        self.arm_home_x = self.get_parameter('arm_home_x').value
        self.arm_home_z = self.get_parameter('arm_home_z').value
        self.arm_grab_lift_x = self.get_parameter('arm_grab_lift_x').value
        self.arm_grab_lift_z = self.get_parameter('arm_grab_lift_z').value
        self.gripper_default_power = self.get_parameter('gripper_default_power').value

        # ---------- Action Clients ----------
        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.arm_client = ActionClient(
            self, MoveArm, '/move_arm', callback_group=self.action_cb_group)
        self.gripper_client = ActionClient(
            self, GripperAction, '/gripper', callback_group=self.action_cb_group)

        # ---------- Service Servers ----------
        self.create_service(ArmMove, '/tennis/arm_move', self._arm_move_cb)
        self.create_service(GripperControl, '/tennis/gripper_control', self._gripper_control_cb)
        self.create_service(Trigger, '/tennis/arm_home', self._arm_home_cb)

        self.get_logger().info('Manipulation Node 启动成功')
        self.get_logger().info('等待 /move_arm 和 /gripper action servers...')

    # ==================================================
    #  Action 通用工具
    # ==================================================
    def _spin_future(self, future, timeout):
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.05)

    def _send_action_blocking(self, client, goal, timeout=30.0, name='action'):
        """发 goal 并阻塞等结果"""
        if not client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn(f'{name} server 不可达')
            return False
        send_future = client.send_goal_async(goal)
        self._spin_future(send_future, timeout)
        if not send_future.done() or not send_future.result():
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().warn(f'{name} goal 被拒绝')
            return False
        result_future = handle.get_result_async()
        self._spin_future(result_future, timeout)
        return result_future.done()

    # ==================================================
    #  Service Callbacks
    # ==================================================
    def _arm_move_cb(self, request, response):
        """Service: 移动机械臂到指定位置"""
        self.get_logger().info(
            f'>>> 机械臂移动: x={request.x:.3f}, z={request.z:.3f}, relative={request.relative}')
        goal = MoveArm.Goal()
        goal.x = float(request.x)
        goal.z = float(request.z)
        goal.relative = bool(request.relative)
        success = self._send_action_blocking(
            self.arm_client, goal, timeout=8.0, name='move_arm')
        response.success = success
        response.message = '机械臂移动完成' if success else '机械臂移动失败'
        return response

    def _gripper_control_cb(self, request, response):
        """Service: 控制夹爪"""
        target = request.target.lower()
        power = request.power if request.power > 0 else self.gripper_default_power
        self.get_logger().info(f'>>> 夹爪控制: {target}, power={power}')

        goal = GripperAction.Goal()
        if target == 'open':
            goal.target_state = GRIPPER_OPEN
        elif target == 'close':
            goal.target_state = GRIPPER_CLOSE
        else:
            goal.target_state = GRIPPER_PAUSE
        goal.power = float(power)

        success = self._send_action_blocking(
            self.gripper_client, goal, timeout=4.0, name='gripper')
        response.success = success
        response.message = f'夹爪 {target} 完成' if success else f'夹爪 {target} 失败'
        return response

    def _arm_home_cb(self, request, response):
        """Service: 机械臂归位"""
        self.get_logger().info('>>> 机械臂归位')
        goal = MoveArm.Goal()
        goal.x = float(self.arm_home_x)
        goal.z = float(self.arm_home_z)
        goal.relative = False
        success = self._send_action_blocking(
            self.arm_client, goal, timeout=8.0, name='move_arm')
        response.success = success
        response.message = '机械臂归位完成' if success else '机械臂归位失败'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TennisManipulationNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
