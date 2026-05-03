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
tennis_fsm_node.py
状态机主控节点 —— 协调视觉、导航、操作节点完成网球抓取任务

订阅:
  /tennis/detections        (vision_msgs/Detection2DArray)  视觉检测结果
  /tennis/pose              (geometry_msgs/PoseStamped)     当前 SLAM 位姿
  /tennis/nav_status        (std_msgs/String)               导航状态
  /tennis/annotated_image   (sensor_msgs/Image, 可选)        调试图像

发布:
  /tennis/nav_goal          (geometry_msgs/PoseStamped)     导航目标
  /tennis/cmd_vel_raw       (geometry_msgs/Twist)           原始速度指令

服务客户端:
  /tennis/arm_move          (tennis_pick/srv/ArmMove)
  /tennis/gripper_control   (tennis_pick/srv/GripperControl)
  /tennis/arm_home          (std_srvs/srv/Trigger)
  /tennis/stop_nav          (std_srvs/srv/Trigger)

状态机: SEARCHING -> ROTATING -> GRABBING -> PLACING -> HOMING -> SEARCHING
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from tennis_pick.srv import ArmMove, GripperControl
from std_srvs.srv import Trigger

import cv2
import numpy as np

# 常量
IMG_W, IMG_H = 640, 360
IMG_CENTER_X = IMG_W // 2
ROTATE_TOLERANCE_PX = 30
DETECT_CONFIRM_FRAMES = 3
LOST_TRIGGER_FRAMES = 5
TOF_TRIGGER_M = 0.08

ARM_HOME_X, ARM_HOME_Z = 0.18, -0.07
ARM_GRAB_LIFT_X, ARM_GRAB_LIFT_Z = 0.08, 0.11


class TennisFSMNode(Node):
    def __init__(self):
        super().__init__('tennis_fsm')

        # ---------- 参数 ----------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('img_w', IMG_W),
                ('img_h', IMG_H),
                ('rotate_tolerance_px', ROTATE_TOLERANCE_PX),
                ('detect_confirm_frames', DETECT_CONFIRM_FRAMES),
                ('lost_trigger_frames', LOST_TRIGGER_FRAMES),
                ('tof_trigger_m', TOF_TRIGGER_M),
                ('arm_home_x', ARM_HOME_X),
                ('arm_home_z', ARM_HOME_Z),
                ('arm_grab_lift_x', ARM_GRAB_LIFT_X),
                ('arm_grab_lift_z', ARM_GRAB_LIFT_Z),
                ('headless', True),
                ('use_dynamic_box_pose', True),  # 启动时记录当前位置为放置点
                ('box_pose', None),  # 可手动指定 [x, y, yaw_deg]
                ('search_rotate_angle', 15.0),  # 搜索时每次旋转角度
                ('grabbing_linear_speed', 0.18),  # 抓取前进速度
            ]
        )

        self.img_w = self.get_parameter('img_w').value
        self.img_h = self.get_parameter('img_h').value
        self.img_center_x = self.img_w // 2
        self.rotate_tolerance_px = self.get_parameter('rotate_tolerance_px').value
        self.detect_confirm_frames = self.get_parameter('detect_confirm_frames').value
        self.lost_trigger_frames = self.get_parameter('lost_trigger_frames').value
        self.tof_trigger_m = self.get_parameter('tof_trigger_m').value
        self.arm_home_x = self.get_parameter('arm_home_x').value
        self.arm_home_z = self.get_parameter('arm_home_z').value
        self.arm_grab_lift_x = self.get_parameter('arm_grab_lift_x').value
        self.arm_grab_lift_z = self.get_parameter('arm_grab_lift_z').value
        self.headless = self.get_parameter('headless').value
        self.use_dynamic_box_pose = self.get_parameter('use_dynamic_box_pose').value
        self.search_rotate_angle = self.get_parameter('search_rotate_angle').value
        self.grabbing_linear_speed = self.get_parameter('grabbing_linear_speed').value

        # box_pose 处理
        box_pose_param = self.get_parameter('box_pose').value
        if box_pose_param is not None and len(box_pose_param) == 3:
            self.box_pose = tuple(box_pose_param)
            self.get_logger().info(f'使用手动指定放置点: {self.box_pose}')
        else:
            self.box_pose = None

        # ---------- 线程锁与缓存 ----------
        self._det_lock = threading.Lock()
        self._latest_detections = []
        self._latest_det_time = 0.0

        self._pose_lock = threading.Lock()
        self._latest_pose = None  # (x, y, yaw_deg)

        self._nav_status_lock = threading.Lock()
        self._nav_status = 'idle'

        self._img_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_time = 0.0

        self.bridge = CvBridge()

        # ---------- 状态机变量 ----------
        self.current_state = 'INIT'
        self.detect_count = 0
        self.lost_count = 0
        self.target_bbox = None
        self.running = True
        self.ctrl_thread = None

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------- 订阅 ----------
        self.create_subscription(
            Detection2DArray, '/tennis/detections', self._detections_cb, qos)
        self.create_subscription(
            PoseStamped, '/tennis/pose', self._pose_cb, 10)
        self.create_subscription(
            String, '/tennis/nav_status', self._nav_status_cb, 10)
        if not self.headless:
            self.create_subscription(
                Image, '/tennis/annotated_image', self._image_cb, qos)

        # ---------- 发布 ----------
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/tennis/nav_goal', 10)
        self.cmd_vel_raw_pub = self.create_publisher(Twist, '/tennis/cmd_vel_raw', 10)

        # ---------- Service Clients ----------
        self.arm_move_cli = self.create_client(ArmMove, '/tennis/arm_move')
        self.gripper_cli = self.create_client(GripperControl, '/tennis/gripper_control')
        self.arm_home_cli = self.create_client(Trigger, '/tennis/arm_home')
        self.stop_nav_cli = self.create_client(Trigger, '/tennis/stop_nav')

        self.get_logger().info('FSM Node 初始化完成，等待服务...')

    # ==================================================
    #  订阅回调
    # ==================================================
    def _detections_cb(self, msg):
        dets = []
        for det in msg.detections:
            if not det.results:
                continue
            bbox = det.bbox
            x1 = int(bbox.center.position.x - bbox.size_x / 2)
            y1 = int(bbox.center.position.y - bbox.size_y / 2)
            x2 = int(bbox.center.position.x + bbox.size_x / 2)
            y2 = int(bbox.center.position.y + bbox.size_y / 2)
            dets.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': det.results[0].hypothesis.score,
                'class_name': det.results[0].hypothesis.class_id,
            })
        with self._det_lock:
            self._latest_detections = dets
            self._latest_det_time = time.time()

    def _pose_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        # 四元数转 yaw
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        yaw_deg = math.degrees(yaw)
        with self._pose_lock:
            self._latest_pose = (x, y, yaw_deg)

    def _nav_status_cb(self, msg):
        with self._nav_status_lock:
            self._nav_status = msg.data

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._img_lock:
                self._latest_frame = frame
                self._latest_frame_time = time.time()
        except Exception as e:
            self.get_logger().warn(f'image_cb: {e}')

    # ==================================================
    #  数据获取
    # ==================================================
    def _get_detections(self):
        with self._det_lock:
            return list(self._latest_detections)

    def _get_pose(self):
        with self._pose_lock:
            return self._latest_pose

    def _get_nav_status(self):
        with self._nav_status_lock:
            return self._nav_status

    def _get_frame(self):
        with self._img_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _wait_nav_complete(self, timeout=60.0):
        """阻塞等待导航完成"""
        start = time.time()
        while time.time() - start < timeout:
            status = self._get_nav_status()
            if status in ('success', 'failed', 'obstacle'):
                return status
            time.sleep(0.2)
        return 'timeout'

    # ==================================================
    #  Service 调用工具
    # ==================================================
    def _call_service(self, client, request, timeout=5.0, name='service'):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f'{name} 服务不可用')
            return None
        future = client.call_async(request)
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.05)
        if future.done():
            return future.result()
        return None

    def arm_move(self, x, z, relative=False):
        req = ArmMove.Request()
        req.x = float(x)
        req.z = float(z)
        req.relative = bool(relative)
        resp = self._call_service(self.arm_move_cli, req, timeout=10.0, name='arm_move')
        return resp.success if resp else False

    def gripper_set(self, target='open', power=0.7):
        req = GripperControl.Request()
        req.target = target
        req.power = float(power)
        resp = self._call_service(self.gripper_cli, req, timeout=5.0, name='gripper_control')
        return resp.success if resp else False

    def arm_home(self):
        req = Trigger.Request()
        resp = self._call_service(self.arm_home_cli, req, timeout=10.0, name='arm_home')
        return resp.success if resp else False

    def stop_nav(self):
        req = Trigger.Request()
        resp = self._call_service(self.stop_nav_cli, req, timeout=3.0, name='stop_nav')
        return resp.success if resp else False

    def send_nav_goal(self, x, y, yaw_deg):
        """发送导航目标"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        yaw = math.radians(yaw_deg)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_goal_pub.publish(msg)

    def send_cmd_vel(self, linear_x, angular_z):
        """发送原始速度"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_raw_pub.publish(twist)

    # ==================================================
    #  辅助
    # ==================================================
    def _select_nearest_ball(self, detections):
        tennis = [d for d in detections if d['class_name'] == 'tennis']
        if not tennis:
            return None
        areas = [(d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) for d in tennis]
        return tennis[int(np.argmax(areas))]

    def _small_angle_search(self):
        self.get_logger().info(f'>>> 旋转 {self.search_rotate_angle}° 搜索')
        self.send_nav_goal(0, 0, self.search_rotate_angle)  # 相对旋转通过 nav 处理
        # 或者直接发速度？这里用 nav_goal 让 navigation 节点处理
        # 但 nav_goal 是绝对坐标... 改为直接发 cmd_vel_raw
        twist = Twist()
        twist.angular.z = -0.3 if self.search_rotate_angle > 0 else 0.3
        self.cmd_vel_raw_pub.publish(twist)
        time.sleep(abs(self.search_rotate_angle) / 30.0)  # 粗略时间
        self.send_cmd_vel(0, 0)

    # ==================================================
    #  状态机
    # ==================================================
    def run_searching(self):
        detections = self._get_detections()
        high = [d for d in detections
                if d['class_name'] == 'tennis' and d['confidence'] > 0.7]
        if high:
            self.detect_count += 1
            self.lost_count = 0
            if self.detect_count >= self.detect_confirm_frames:
                self.target_bbox = self._select_nearest_ball(high)['bbox']
                self.detect_count = 0
                self.get_logger().info('>>> SEARCHING: 确认目标 -> ROTATING')
                return 'ROTATING'
        else:
            self.lost_count += 1
            self.detect_count = 0
            if self.lost_count >= self.lost_trigger_frames:
                self.lost_count = 0
                self._small_angle_search()
        time.sleep(0.1)
        return 'SEARCHING'

    def run_rotating(self):
        max_retry = 30
        last_offset = None
        oscillation_count = 0
        damping = 0.7

        for cnt in range(max_retry):
            detections = self._get_detections()
            high = [d for d in detections
                    if d['class_name'] == 'tennis' and d['confidence'] > 0.6]
            if not high:
                self.get_logger().info('>>> ROTATING: 丢失目标 -> SEARCHING')
                return 'SEARCHING'

            nearest = self._select_nearest_ball(high)
            x1, _, x2, _ = nearest['bbox']
            target_center_x = (x1 + x2) / 2
            offset = target_center_x - self.img_center_x

            if abs(offset) <= self.rotate_tolerance_px:
                self.get_logger().info(f'>>> ROTATING: offset={offset:+.0f} ✓ 对准完成')
                self.target_bbox = nearest['bbox']
                return 'GRABBING'

            # 振荡检测
            if last_offset is not None and last_offset * offset < 0:
                oscillation_count += 1
                damping *= 0.6
                self.get_logger().warn(
                    f'>>> ROTATING: 振荡 #{oscillation_count} damping={damping:.2f}')
            last_offset = offset

            angle = (offset / (self.img_w / 2)) * 50 * damping
            angle = max(-15, min(15, angle))
            if 0 < abs(angle) < 1.5:
                angle = 1.5 if angle > 0 else -1.5

            self.get_logger().info(f'>>> ROTATING: offset={offset:+.0f} -> 旋转 {angle:+.1f}°')

            # 发送旋转命令（通过 nav_goal 或直接 cmd_vel）
            # 这里用 cmd_vel_raw 让 navigation 做安全处理
            # 旋转速度约 0.3 rad/s，转 angle 度需要时间
            twist = Twist()
            twist.angular.z = -0.3 if angle > 0 else 0.3
            self.cmd_vel_raw_pub.publish(twist)
            rotate_time = abs(math.radians(angle)) / 0.3
            time.sleep(rotate_time)
            self.send_cmd_vel(0, 0)
            time.sleep(0.5)  # 停稳

        self.get_logger().info('>>> ROTATING: 次数耗尽 -> SEARCHING')
        return 'SEARCHING'

    def run_grabbing(self):
        self.get_logger().info('>>> GRABBING: 打开夹爪')
        self.gripper_set(target='open', power=0.8)

        self.get_logger().info('>>> GRABBING: 前进逼近 (边走边修正)')
        start = time.time()
        arrived = False

        while time.time() - start < 10:
            # 读图像做角度修正
            detections = self._get_detections()
            angular_correction = 0.0
            if detections:
                high = [det for det in detections
                        if det['class_name'] == 'tennis' and det['confidence'] > 0.5]
                if high:
                    nearest = self._select_nearest_ball(high)
                    x1, _, x2, _ = nearest['bbox']
                    cx = (x1 + x2) / 2
                    offset = cx - self.img_center_x
                    angular_correction = -offset / 320.0 * 0.3
                    angular_correction = max(-0.5, min(0.5, angular_correction))

            self.send_cmd_vel(self.grabbing_linear_speed, angular_correction)
            time.sleep(0.1)

        # 停车
        self.send_cmd_vel(0, 0)
        time.sleep(0.3)

        if not arrived:
            # 没有 ToF 数据，靠时间/视觉判断，这里简化处理
            self.get_logger().warn('>>> GRABBING: 时间到，尝试抓取')

        # 闭合夹爪
        self.get_logger().info('>>> GRABBING: 闭合夹爪')
        self.gripper_set(target='close', power=0.6)
        time.sleep(0.3)

        # 抬起机械臂
        self.get_logger().info('>>> GRABBING: 抬起机械臂')
        self.arm_move(self.arm_grab_lift_x, self.arm_grab_lift_z, relative=False)
        return 'PLACING'

    def run_placing(self):
        self.get_logger().info('>>> PLACING: SLAM 导航到箱子')
        pose = self._get_pose()
        if pose is None:
            self.get_logger().error('>>> PLACING: 无 SLAM 位姿 -> HOMING')
            return 'HOMING'
        self.get_logger().info(f'  当前位姿 ({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.0f}°)')

        if self.box_pose is None:
            self.get_logger().error('>>> PLACING: 未设置放置点 -> HOMING')
            return 'HOMING'

        # 发送导航目标
        self.send_nav_goal(*self.box_pose)
        status = self._wait_nav_complete(timeout=60.0)

        if status != 'success':
            self.get_logger().warn(f'>>> PLACING: 导航失败({status}) -> HOMING')
            return 'HOMING'

        self.get_logger().info('>>> PLACING: 松开夹爪放球')
        self.gripper_set(target='open', power=0.8)
        time.sleep(0.5)

        self.get_logger().info('>>> PLACING: 后退 50cm')
        self.send_cmd_vel(-0.15, 0)
        time.sleep(3.5)  # 后退约 50cm
        self.send_cmd_vel(0, 0)

        self.get_logger().info('>>> PLACING: 旋转 180°')
        twist = Twist()
        twist.angular.z = 0.5
        self.cmd_vel_raw_pub.publish(twist)
        time.sleep(math.pi / 0.5)  # 180° = π rad
        self.send_cmd_vel(0, 0)

        return 'HOMING'

    def run_homing(self):
        self.get_logger().info('>>> HOMING')
        self.arm_home()
        self.gripper_set(target='open', power=0.8)
        self.detect_count = 0
        self.lost_count = 0
        self.get_logger().info('>>> HOMING: 完成')
        return 'SEARCHING'

    # ==================================================
    #  控制循环
    # ==================================================
    def control_loop(self):
        self.get_logger().info('>>> 控制循环启动')
        time.sleep(1.5)

        # 初始化机械臂和夹爪
        self.arm_home()
        self.gripper_set(target='open', power=0.8)

        # 动态记录放置点
        if self.use_dynamic_box_pose and self.box_pose is None:
            self.get_logger().info('>>> 正在等待 SLAM 初始化并记录起点...')
            for _ in range(20):
                pose = self._get_pose()
                if pose is not None:
                    self.box_pose = pose
                    self.get_logger().info(
                        f'>>> 成功记录起点 (BOX_POSE): x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.0f}°')
                    break
                time.sleep(0.5)
            if self.box_pose is None:
                self.get_logger().error('>>> 无法获取 SLAM 初始位姿，采用默认 (0,0,0)')
                self.box_pose = (0.0, 0.0, 0.0)

        self.current_state = 'SEARCHING'

        while self.running and rclpy.ok():
            try:
                if self.current_state == 'SEARCHING':
                    self.current_state = self.run_searching()
                elif self.current_state == 'ROTATING':
                    self.current_state = self.run_rotating()
                elif self.current_state == 'GRABBING':
                    self.current_state = self.run_grabbing()
                elif self.current_state == 'PLACING':
                    self.current_state = self.run_placing()
                elif self.current_state == 'HOMING':
                    self.current_state = self.run_homing()
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f'状态机异常: {e}')
                import traceback
                traceback.print_exc()
                self.current_state = 'HOMING'
                time.sleep(1.0)

        self.get_logger().info('控制线程退出')

    def show_loop(self):
        if self.headless:
            self.get_logger().info('HEADLESS 模式：不显示画面，按 Ctrl+C 退出')
            try:
                while self.running and rclpy.ok():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.running = False
            return

        try:
            while self.running and rclpy.ok():
                frame = self._get_frame()
                if frame is not None:
                    pose = self._get_pose()
                    pose_str = f'({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.0f})' if pose else 'N/A'
                    cv2.putText(frame, f'STATE: {self.current_state}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f'SLAM: {pose_str}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                    cv2.imshow('Tennis Pick FSM', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.running = False
                        break
                else:
                    cv2.waitKey(50)
        finally:
            cv2.destroyAllWindows()

    def start(self):
        self.ctrl_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.ctrl_thread.start()

    def shutdown(self):
        self.running = False
        if self.ctrl_thread:
            self.ctrl_thread.join(timeout=2.0)
        self.stop_nav()
        self.send_cmd_vel(0, 0)


def main(args=None):
    rclpy.init(args=args)
    node = TennisFSMNode()
    node.start()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.show_loop()
    except KeyboardInterrupt:
        node.get_logger().info('用户中断')
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
