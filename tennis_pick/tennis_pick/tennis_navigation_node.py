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
tennis_navigation_node.py
导航/底盘节点 —— 负责 SLAM 位姿、底盘移动、ToF 避障

订阅:
  /range_0                  (sensor_msgs/Range)        ToF 测距
  /tennis/nav_goal          (geometry_msgs/PoseStamped) 导航目标 (x,y,yaw)
  /tennis/cmd_vel_raw       (geometry_msgs/Twist)       FSM 原始速度指令

发布:
  /cmd_vel                  (geometry_msgs/Twist)       实际底盘速度（经安全处理）
  /tennis/pose              (geometry_msgs/PoseStamped) 当前 SLAM 位姿
  /tennis/nav_status        (std_msgs/String)           导航状态: idle/navigating/success/failed/obstacle

服务:
  /tennis/stop_nav          (std_srvs/Trigger)          停止当前导航
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from robomaster_msgs.action import Move
from tennis_pick.utils import euler_from_quaternion, normalize_angle


# 常量
NAV_POS_TOLERANCE = 0.12
NAV_YAW_TOLERANCE = 3.0
NAV_TIMEOUT = 60.0
NAV_FORWARD_STEP = 0.3
TOF_TRIGGER_M = 0.08
CHASSIS_LINEAR_SPEED = 0.3
CHASSIS_ANGULAR_SPEED = math.radians(45)
OBSTACLE_STOP_DIST = 0.15  # 避障触发距离


class TennisNavigationNode(Node):
    def __init__(self):
        super().__init__('tennis_navigation')

        # ---------- 参数 ----------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tof_topic', '/range_0'),
                ('nav_pos_tolerance', NAV_POS_TOLERANCE),
                ('nav_yaw_tolerance', NAV_YAW_TOLERANCE),
                ('nav_timeout', NAV_TIMEOUT),
                ('nav_forward_step', NAV_FORWARD_STEP),
                ('tof_trigger_m', TOF_TRIGGER_M),
                ('chassis_linear_speed', CHASSIS_LINEAR_SPEED),
                ('chassis_angular_speed_deg', 45.0),
                ('obstacle_stop_dist', OBSTACLE_STOP_DIST),
                ('map_frame', 'map'),
                ('base_frame', 'base_footprint'),
            ]
        )

        self.tof_topic = self.get_parameter('tof_topic').value
        self.nav_pos_tolerance = self.get_parameter('nav_pos_tolerance').value
        self.nav_yaw_tolerance = self.get_parameter('nav_yaw_tolerance').value
        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.nav_forward_step = self.get_parameter('nav_forward_step').value
        self.tof_trigger_m = self.get_parameter('tof_trigger_m').value
        self.linear_speed = self.get_parameter('chassis_linear_speed').value
        self.angular_speed = math.radians(self.get_parameter('chassis_angular_speed_deg').value)
        self.obstacle_stop_dist = self.get_parameter('obstacle_stop_dist').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # ---------- TF ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- ToF ----------
        self._tof_lock = threading.Lock()
        self._tof_m = None

        # ---------- 导航状态 ----------
        self._nav_lock = threading.Lock()
        self._nav_status = 'idle'  # idle, navigating, success, failed, obstacle
        self._nav_stop_flag = False
        self._nav_thread = None

        # ---------- Callback Groups ----------
        self.sub_cb_group = ReentrantCallbackGroup()
        self.action_cb_group = MutuallyExclusiveCallbackGroup()

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------- 订阅 ----------
        self.create_subscription(
            Range, self.tof_topic, self._tof_cb, qos,
            callback_group=self.sub_cb_group)
        self.create_subscription(
            PoseStamped, '/tennis/nav_goal', self._nav_goal_cb, 10,
            callback_group=self.sub_cb_group)
        self.create_subscription(
            Twist, '/tennis/cmd_vel_raw', self._cmd_vel_raw_cb, 10,
            callback_group=self.sub_cb_group)

        # ---------- 发布 ----------
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/tennis/pose', 10)
        self.nav_status_pub = self.create_publisher(String, '/tennis/nav_status', 10)

        # ---------- Service ----------
        self.stop_nav_srv = self.create_service(
            Trigger, '/tennis/stop_nav', self._stop_nav_cb,
            callback_group=self.sub_cb_group)

        # ---------- Action Client ----------
        self.move_client = ActionClient(
            self, Move, '/move', callback_group=self.action_cb_group)

        # ---------- 定时器：发布位姿 ----------
        self.create_timer(0.2, self._publish_pose)
        self.create_timer(0.5, self._publish_nav_status)

        self.get_logger().info('Navigation Node 启动成功')

    # ==================================================
    #  订阅回调
    # ==================================================
    def _tof_cb(self, msg):
        r = msg.range
        if 0 < r < 8.0:
            with self._tof_lock:
                self._tof_m = r

    def _get_tof_m(self):
        with self._tof_lock:
            return self._tof_m

    def _cmd_vel_raw_cb(self, msg):
        """FSM 直接速度控制（如 GRABBING 阶段），经安全处理后下发"""
        safe_twist = Twist()
        safe_twist.linear.x = msg.linear.x
        safe_twist.angular.z = msg.angular.z

        # 避障：如果前进方向有障碍物，禁止前进
        tof = self._get_tof_m()
        if tof is not None and tof < self.obstacle_stop_dist and msg.linear.x > 0:
            safe_twist.linear.x = 0.0
            self.get_logger().warn(f'避障触发! ToF={tof:.2f}m, 禁止前进')

        self.cmd_vel_pub.publish(safe_twist)

    def _nav_goal_cb(self, msg):
        """收到导航目标，启动导航线程"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        # yaw 从四元数提取，或直接用 position.z 存储（约定）
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = math.degrees(yaw)

        self.get_logger().info(f'收到导航目标: ({x:.2f}, {y:.2f}, {yaw_deg:.1f}°)')

        # 如果已有导航在进行，先停止
        self._request_stop()
        if self._nav_thread and self._nav_thread.is_alive():
            self._nav_thread.join(timeout=2.0)

        self._nav_stop_flag = False
        self._set_nav_status('navigating')
        self._nav_thread = threading.Thread(
            target=self._navigate_worker, args=(x, y, yaw_deg), daemon=True)
        self._nav_thread.start()

    def _stop_nav_cb(self, request, response):
        """Service: 停止导航"""
        self._request_stop()
        response.success = True
        response.message = '导航停止指令已发送'
        return response

    def _request_stop(self):
        with self._nav_lock:
            self._nav_stop_flag = True

    def _should_stop(self):
        with self._nav_lock:
            return self._nav_stop_flag

    def _set_nav_status(self, status):
        with self._nav_lock:
            self._nav_status = status

    def _get_nav_status(self):
        with self._nav_lock:
            return self._nav_status

    # ==================================================
    #  位姿发布
    # ==================================================
    def _publish_pose(self):
        pose = self.get_slam_pose()
        if pose is None:
            return
        x, y, yaw_deg = pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        # yaw 转四元数 (只绕 Z 轴)
        yaw = math.radians(yaw_deg)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(msg)

    def _publish_nav_status(self):
        msg = String()
        msg.data = self._get_nav_status()
        self.nav_status_pub.publish(msg)

    # ==================================================
    #  SLAM 位姿
    # ==================================================
    def get_slam_pose(self, timeout=0.1):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, math.degrees(yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ==================================================
    #  Action 工具
    # ==================================================
    def _spin_future(self, future, timeout):
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.05)

    def _send_move_action(self, x=0.0, y=0.0, theta_deg=0.0,
                          linear_speed=None, angular_speed_deg=None,
                          timeout=20.0):
        """发送 /move action 并阻塞等待结果"""
        if linear_speed is None:
            linear_speed = self.linear_speed
        if angular_speed_deg is None:
            angular_speed_deg = math.degrees(self.angular_speed)

        if not self.move_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn('/move action server 不可达')
            return False

        goal = Move.Goal()
        goal.x = float(x)
        goal.y = float(y)
        goal.theta = float(math.radians(theta_deg))
        goal.linear_speed = float(linear_speed)
        goal.angular_speed = float(math.radians(angular_speed_deg))

        send_future = self.move_client.send_goal_async(goal)
        self._spin_future(send_future, timeout)
        if not send_future.done() or not send_future.result():
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().warn('/move goal 被拒绝')
            return False
        result_future = handle.get_result_async()
        self._spin_future(result_future, timeout)
        return result_future.done()

    # ==================================================
    #  导航工作线程（核心）
    # ==================================================
    def _navigate_worker(self, tx, ty, tyaw_deg):
        """在独立线程中执行导航"""
        start = time.time()
        success = False

        try:
            # ---- 阶段 1: 转向目标位置 ----
            self.get_logger().info('>>> NAV-1 转向目标位置')
            for _ in range(15):
                if self._should_stop():
                    break
                if time.time() - start > self.nav_timeout:
                    break
                pose = self.get_slam_pose()
                if pose is None:
                    time.sleep(0.3)
                    continue
                cx, cy, cyaw = pose
                dx, dy = tx - cx, ty - cy
                if math.hypot(dx, dy) < self.nav_pos_tolerance:
                    break
                target_heading = math.degrees(math.atan2(dy, dx))
                dyaw = normalize_angle(target_heading - cyaw)
                if abs(dyaw) < self.nav_yaw_tolerance:
                    break
                self.get_logger().info(f'  转 {dyaw:+.1f}°')
                self._send_move_action(0, 0, dyaw, angular_speed_deg=20)
                time.sleep(0.2)

            # ---- 阶段 2: 直线前进 ----
            self.get_logger().info('>>> NAV-2 前进')
            dist = 999999.0
            dist_last = 99999.0
            for _ in range(15):
                if self._should_stop():
                    break
                if time.time() - start > self.nav_timeout:
                    break
                pose = self.get_slam_pose()
                if pose is None:
                    time.sleep(0.3)
                    continue
                cx, cy, _ = pose
                dist_last = dist
                dist = math.hypot(tx - cx, ty - cy)
                if dist_last < dist:
                    break
                if dist < self.nav_pos_tolerance:
                    break
                # 避障检查
                tof = self._get_tof_m()
                if tof is not None and tof < self.obstacle_stop_dist:
                    self.get_logger().warn(f'>>> NAV 避障停止! ToF={tof:.2f}m')
                    self._set_nav_status('obstacle')
                    return
                step = min(self.nav_forward_step, dist - self.nav_pos_tolerance / 2)
                self.get_logger().info(f'  前进 {step:.2f}m (剩余 {dist:.2f}m)')
                self._send_move_action(step, 0, 0)

            # ---- 阶段 3: 调最终朝向 ----
            self.get_logger().info(f'>>> NAV-3 朝向 {tyaw_deg:.0f}°')
            for _ in range(8):
                if self._should_stop():
                    break
                if time.time() - start > self.nav_timeout:
                    break
                pose = self.get_slam_pose()
                if pose is None:
                    time.sleep(0.3)
                    continue
                _, _, cyaw = pose
                dyaw = normalize_angle(tyaw_deg - cyaw)
                if abs(dyaw) < self.nav_yaw_tolerance:
                    self.get_logger().info(f'>>> NAV 到达 yaw={cyaw:.0f}°')
                    success = True
                    break
                self._send_move_action(0, 0, dyaw, angular_speed_deg=30)

            # 最终检查
            if not success:
                pose = self.get_slam_pose()
                if pose:
                    cx, cy, cyaw = pose
                    dist = math.hypot(tx - cx, ty - cy)
                    if dist < self.nav_pos_tolerance * 1.5:
                        self.get_logger().info('>>> NAV 接近到位（容差放宽）')
                        success = True

        except Exception as e:
            self.get_logger().error(f'导航异常: {e}')

        if success:
            self._set_nav_status('success')
        else:
            self._set_nav_status('failed')

    def stop_chassis(self):
        """发送停止命令"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TennisNavigationNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_chassis()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
