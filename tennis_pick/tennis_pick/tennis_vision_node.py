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
tennis_vision_node.py
视觉识别节点 —— 负责 YOLO 网球检测

架构参考: BrickPick vision_node
订阅: /camera/image_color
发布: /tennis/detections (vision_msgs/Detection2DArray)
       /tennis/annotated_image (sensor_msgs/Image, 调试用)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import os
import time

# YOLOv5 依赖 (需将 yolov5 加入 PYTHONPATH 或在同目录)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


class TennisVisionNode(Node):
    def __init__(self):
        super().__init__('tennis_vision')

        # ---------- 参数声明 ----------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', './best.pt'),
                ('camera_topic', '/camera/image_color'),
                ('conf_threshold', 0.5),
                ('iou_threshold', 0.45),
                ('imgsz', 640),
                ('publish_debug_image', True),
                ('device', ''),  # 空字符串自动选择 cuda/cpu
                ('target_class', 'tennis'),  # 目标类别名
            ]
        )

        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        device_str = self.get_parameter('device').value
        self.target_class = self.get_parameter('target_class').value

        # ---------- YOLO 初始化 ----------
        self.device = torch.device(
            device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.get_logger().info(f'正在加载 YOLO 模型: {model_path} 在设备 {self.device} 上...')

        if not os.path.exists(model_path):
            self.get_logger().error(f'模型文件不存在: {model_path}')
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

        self.model = DetectMultiBackend(model_path, device=self.device)
        self.stride, self.names = self.model.stride, self.model.names
        self.get_logger().info(f'YOLO 加载完成。类别: {self.names}')

        self.bridge = CvBridge()

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------- 订阅与发布 ----------
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, qos)

        self.detection_pub = self.create_publisher(
            Detection2DArray, '/tennis/detections', 10)

        if self.publish_debug:
            self.image_pub = self.create_publisher(
                Image, '/tennis/annotated_image', 10)

        self.get_logger().info(f'Vision Node 启动成功，监听话题: {camera_topic}')

    def image_callback(self, msg):
        try:
            # 1. 图像转换
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 2. YOLO 推理
            detections = self._yolo_detect(cv_image)

            # 3. 构建 Detection2DArray
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            for det in detections:
                detection = Detection2D()
                detection.header = msg.header

                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                sx = x2 - x1
                sy = y2 - y1

                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.size_x = float(sx)
                detection.bbox.size_y = float(sy)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = det['class_name']
                hyp.hypothesis.score = float(det['confidence'])
                detection.results.append(hyp)

                detection_array.detections.append(detection)

            # 4. 发布检测结果
            self.detection_pub.publish(detection_array)

            # 5. 调试图像
            if self.publish_debug:
                annotated = self._draw(cv_image, detections)
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                out_msg.header = msg.header
                self.image_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'处理图像时发生错误: {e}')

    def _yolo_detect(self, frame):
        """YOLOv5 推理"""
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)
            pred = non_max_suppression(
                pred, self.conf_threshold, self.iou_threshold)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    class_name = self.names[int(cls)]
                    if class_name == self.target_class:
                        detections.append({
                            'bbox': [int(x) for x in xyxy],
                            'confidence': float(conf),
                            'class_name': class_name,
                        })
        return detections

    def _draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class_name']}:{det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


def main(args=None):
    rclpy.init(args=args)
    node = TennisVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
