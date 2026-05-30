# ==========================================================
#  tennis_pick_nav2.py
#  网球识别抓取 + Nav2 自动定位/避障放置 (ROS2 Humble)：
#    - PLACING 阶段：
#      改用 Nav2 的 NavigateToPose action。Nav2 自动做：
#        * 全局路径规划（A*/Dijkstra）
#        * 局部避障（DWB + costmap）
#        * 到达后用激光匹配做精细对齐
#    - 启动时会往 /initialpose 推一次当前 TF 位姿，让 AMCL 收敛
#    - 箱子位置可通过 ros2 参数 box_x / box_y / box_yaw_deg 指定，
#      不指定则沿用旧版"启动位置即箱子"逻辑作为兜底
#    - 其他状态（SEARCHING / ROTATING / GRABBING / HOMING）逻辑
#  状态机：SEARCHING -> ROTATING -> GRABBING -> PLACING -> HOMING -> SEARCHING
# ==========================================================
import math
import threading
import time
import random
import json
import queue as queue_mod
import multiprocessing as mp

import cv2
import numpy as np
import torch

# ---------- 语音识别相关（Vosk + ReSpeaker） ----------
# 安装：
#   pip install vosk sounddevice
#   下载中文小模型：https://alphacephei.com/vosk/models
#     vosk-model-small-cn-0.22.zip，解压后放到 VOSK_MODEL_PATH 指向的目录
#
# ★★★ 关键修改：sounddevice / vosk 的 import 推迟到子进程内部 ★★★
# 这样主进程的 YOLO/Nav2 不会被 PortAudio 的 ALSA 初始化干扰；
# 同时子进程拥有独立 GIL，音频采集和识别完全不会被主进程卡住。

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Quaternion
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from cv_bridge import CvBridge

# robomaster_ros action（保留：抓取阶段精确小位移用）
from robomaster_msgs.action import MoveArm, GripperControl, Move

# Nav2 action
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

# YOLOv5
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


HEADLESS = True  # 远程 SSH 跑设 True，本地有显示器跑设 False

# ==================================================
#  配置常量
# ==================================================
# 图像
IMG_W, IMG_H = 640, 360
IMG_CENTER_X = IMG_W // 2
ROTATE_TOLERANCE_PX = 30

# 机械臂姿态
ARM_HOME_X, ARM_HOME_Z = 0.18, -0.07
ARM_GRAB_LIFT_X, ARM_GRAB_LIFT_Z = 0.08, 0.11

# 检测去抖
DETECT_CONFIRM_FRAMES = 3
LOST_TRIGGER_FRAMES = 5

# ToF 触发距离
TOF_TRIGGER_M = 0.13

# 抓取阶段速度
CHASSIS_LINEAR_SPEED = 0.3
CHASSIS_ANGULAR_SPEED = math.radians(45)

# Nav2 相关（新增）
NAV2_ACTION_NAME = '/navigate_to_pose'
NAV2_TIMEOUT_SEC = 90.0           # 单次导航最长允许时间（含 Nav2 内部 recovery）
NAV2_MAX_RETRY = 2                 # 失败后最多再尝试几次
INITIAL_POSE_TOPIC = '/initialpose'

# GripperControl 状态常量
GRIPPER_PAUSE = 0
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 2

# ==================================================
#  语音识别配置
# ==================================================
VOSK_MODEL_PATH = './vosk-model-small-cn-0.22'   # 中文小模型解压目录
VOICE_SAMPLE_RATE = 16000                          # ReSpeaker 默认 16kHz
VOICE_BLOCK_SIZE = 1024                            # 64ms @16kHz, 让 Vosk 频繁吃数据
VOICE_DEVICE_HINT = 'ReSpeaker'                    # 优先匹配名字含此关键词的设备
# ★ 软件增益：把音频信号放大若干倍，再喂给 Vosk
VOICE_GAIN = 8.0
# 词表（grammar）：每个候选词单独一项！写成 "停 开始" 会被当成一个短语
# 加进若干同音/近音字，提升 Vosk 小模型对短词的鲁棒性
VOICE_GRAMMAR = ('["停", "停下", "停止", "停车", '
                 '"开始", "继续", "走", "出发", '
                 '"[unk]"]')
# 触发词
PAUSE_WORDS = {'停', '停下', '停止', '停车'}
RESUME_WORDS = {'开始', '继续', '走', '出发'}


def euler_from_quaternion(quaternion):
    """四元数 [x, y, z, w] -> [roll, pitch, yaw]"""
    x, y, z, w = quaternion
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_from_yaw(yaw_rad):
    """只绕 Z 轴旋转的四元数"""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


# ==================================================
#  语音控制器 (子进程版)
# ==================================================
#
# 子进程版的解决方案：
#   把音频采集 + Vosk 识别整个搬到独立 multiprocessing.Process。
#   子进程有独立的 GIL 和调度，不受 YOLO/Nav2 卡顿影响。
#   主进程只通过一个轻量级 Queue 接收 'pause' / 'resume' 字符串，
#   开销几乎为 0。
# ==================================================


def _voice_worker(model_path, sample_rate, block_size, device_hint,
                  grammar, pause_words, resume_words, gain,
                  event_q, ready_evt, stop_evt):
    """
    子进程入口。运行在独立进程里，不受主进程 GIL 影响。

    参数全部走值传递（pickle）；sounddevice / vosk 在这里才 import，
    避免主进程一启动就触发 ALSA 探测、把日志刷得满屏都是。

    通信：
      - event_q : mp.Queue, 子进程 -> 主进程，写入 'pause' / 'resume' / 'error:xxx'
      - ready_evt : mp.Event, 子进程初始化完成后 set()
      - stop_evt : mp.Event, 主进程置位后子进程退出
      - gain : float, 软件增益，原始信号乘以这个数（带削峰）
    """
    import sys
    import time
    import json
    import queue as _queue_mod  # 进程内本地 queue（给 sounddevice 回调用）
    import numpy as np

    # 子进程内才 import 音频库
    try:
        import sounddevice as sd
        from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
    except Exception as e:
        try:
            event_q.put_nowait(f'error:import failed: {e}')
        except Exception:
            pass
        ready_evt.set()
        return

    def _log(msg):
        # 子进程直接 print 到 stderr
        print(f'[Voice-PROC] {msg}', file=sys.stderr, flush=True)

    # --------- 选设备 ---------
    device_index = None
    try:
        devices = sd.query_devices()
        hint_lower = device_hint.lower()
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0 and hint_lower in dev['name'].lower():
                device_index = idx
                break
    except Exception as e:
        _log(f'query_devices 失败: {e}')

    if device_index is not None:
        try:
            info = sd.query_devices(device_index)
            _log(f"用设备 #{device_index}: {info['name']}")
        except Exception:
            pass
    else:
        _log('未找到 ReSpeaker，使用系统默认输入设备')

    # --------- 加载模型 ---------
    _log(f'加载 Vosk 模型: {model_path}')
    t0 = time.time()
    try:
        model = VoskModel(model_path)
    except Exception as e:
        try:
            event_q.put_nowait(f'error:vosk model load failed: {e}')
        except Exception:
            pass
        ready_evt.set()
        return
    recognizer = KaldiRecognizer(model, sample_rate, grammar)
    recognizer.SetWords(False)
    _log(f'模型加载完成 ({time.time() - t0:.1f}s)')

    # --------- 音频回调（进程内本地队列，不跨进程） ---------
    audio_q = _queue_mod.Queue(maxsize=400)

    # 削峰阈值：float32 域是 ±1.0
    _gain = float(gain) if gain and gain > 0 else 1.0

    def audio_cb(indata, frames, time_info, status):
        if status:
            pass  # input overflow 之类的，不致命
        try:
            # indata: float32 形状 (frames, 1)，范围 -1.0 ~ +1.0
            arr = indata[:, 0]
            if _gain != 1.0:
                arr = np.clip(arr * _gain, -1.0, 1.0)
            # 转 int16 PCM bytes（Vosk 要求）
            pcm = (arr * 32767.0).astype(np.int16).tobytes()
            audio_q.put_nowait(pcm)
        except _queue_mod.Full:
            pass
        except Exception:
            pass

    # --------- 启动音频流 ---------
    # 用 InputStream（不是 Raw），回调收到的 indata 直接是 numpy float32 数组
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            dtype='float32',
            channels=1,
            device=device_index,
            callback=audio_cb,
        )
        stream.start()
    except Exception as e:
        try:
            event_q.put_nowait(f'error:open stream failed: {e}')
        except Exception:
            pass
        ready_evt.set()
        return

    _log(f'音频流已启动 (gain={_gain}x)，开始识别...')
    ready_evt.set()  # 通知主进程

    pause_set = set(pause_words)
    resume_set = set(resume_words)
    last_heartbeat = time.time()
    chunks_seen = 0
    last_partial_print = 0.0
    last_dispatched = ''      # 防止 partial+final 重复触发同一个词
    last_dispatch_time = 0.0  # 同一关键词去重的时间窗口

    def _dispatch(text, is_partial):
        """在子进程里做关键词匹配；命中就往主进程发"""
        nonlocal last_dispatched, last_dispatch_time
        if not text:
            return
        action = None
        word = None
        for w in pause_set:
            if w in text:
                action = 'pause'; word = w; break
        if action is None:
            for w in resume_set:
                if w in text:
                    action = 'resume'; word = w; break
        if action is None:
            return

        # 同一动作 1.5 秒内不重复发（避免 partial+final 双触发，或连续帧重复）
        now = time.time()
        key = f'{action}:{word}'
        if key == last_dispatched and (now - last_dispatch_time) < 1.5:
            return
        last_dispatched = key
        last_dispatch_time = now

        tag = '(部分)' if is_partial else '(完整)'
        _log(f">>> 命中 '{word}' {tag} 文本='{text}' -> {action.upper()}")
        try:
            event_q.put_nowait(action)
        except Exception as e:
            _log(f'event_q.put 失败: {e}')

        # partial 触发后 Reset，让识别器重新开始下一段
        if is_partial:
            try:
                recognizer.Reset()
            except Exception:
                pass

    # --------- 主识别循环 ---------
    # 峰值跟踪：在心跳间隔内，记录最大音量，方便用户判断是否需要调增益
    peak_vol_window = 0.0
    try:
        while not stop_evt.is_set():
            try:
                pcm = audio_q.get(timeout=0.5)
            except _queue_mod.Empty:
                now = time.time()
                if now - last_heartbeat > 5.0:
                    _log(f'警告：5秒无音频，麦克风可能挂了 (chunks={chunks_seen})')
                    last_heartbeat = now
                continue

            chunks_seen += 1
            now = time.time()

            # 每块都更新峰值
            try:
                arr = np.frombuffer(pcm, dtype=np.int16)
                cur_vol = float(np.abs(arr).mean())
                if cur_vol > peak_vol_window:
                    peak_vol_window = cur_vol
            except Exception:
                cur_vol = 0.0

            if now - last_heartbeat > 5.0:
                # 显示窗口内峰值音量
                _log(f'心跳 chunks={chunks_seen} 峰值音量={peak_vol_window:.0f} '
                     f'(说话期望>3000) 队列堆积={audio_q.qsize()}')
                peak_vol_window = 0.0
                last_heartbeat = now

            if recognizer.AcceptWaveform(pcm):
                result = json.loads(recognizer.Result())
                text = result.get('text', '').replace(' ', '')
                if text:
                    _log(f"FINAL: '{text}'")
                    _dispatch(text, is_partial=False)
            else:
                partial = json.loads(recognizer.PartialResult())
                ptext = partial.get('partial', '').replace(' ', '')
                if ptext:
                    if (now - last_partial_print) > 0.5:
                        _log(f"partial: '{ptext}'")
                        last_partial_print = now
                    _dispatch(ptext, is_partial=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _log(f'识别循环异常: {e}')
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        _log('子进程退出')


class VoiceController:
    """
    用独立子进程跑 Vosk + sounddevice。
    对外接口和旧版线程版完全一样：start() / stop()，
    构造时传 on_pause / on_resume 回调。
    """

    def __init__(self, on_pause, on_resume, logger=None,
                 model_path=VOSK_MODEL_PATH,
                 sample_rate=VOICE_SAMPLE_RATE,
                 block_size=VOICE_BLOCK_SIZE,
                 device_hint=VOICE_DEVICE_HINT):
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.logger = logger
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_hint = device_hint

        # 跨进程通信对象
        # 用 'spawn' 上下文，避免 fork 出去的子进程继承一堆 ROS / CUDA 的句柄
        # （fork CUDA 上下文会让子进程一启动就崩）
        self._mp_ctx = mp.get_context('spawn')
        self._event_q = self._mp_ctx.Queue(maxsize=64)
        self._ready_evt = self._mp_ctx.Event()
        self._stop_evt = self._mp_ctx.Event()
        self._proc = None

        # 主进程内的 dispatcher 线程：从 mp.Queue 读事件 -> 调回调
        self._dispatcher_thread = None
        self._running = False

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def start(self):
        if self._running:
            return
        self._running = True

        # 词表参数转成可 pickle 的 list/str
        grammar = VOICE_GRAMMAR
        pause_words = list(PAUSE_WORDS)
        resume_words = list(RESUME_WORDS)

        self._log('>>> [Voice] 启动子进程...')
        self._proc = self._mp_ctx.Process(
            target=_voice_worker,
            args=(self.model_path, self.sample_rate, self.block_size,
                  self.device_hint, grammar, pause_words, resume_words,
                  VOICE_GAIN,
                  self._event_q, self._ready_evt, self._stop_evt),
            daemon=True,
            name='VoiceWorker',
        )
        self._proc.start()

        # 等子进程初始化完成（模型加载 + 打开音频流），最多 30 秒
        if not self._ready_evt.wait(timeout=30.0):
            self._log('>>> [Voice] 子进程初始化超时！')
        else:
            self._log('>>> [Voice] 子进程已就绪')

        # 排空一下错误队列，看子进程有没有报错
        try:
            while True:
                msg = self._event_q.get_nowait()
                if isinstance(msg, str) and msg.startswith('error:'):
                    self._log(f'>>> [Voice] 子进程报错: {msg[6:]}')
                else:
                    # 不应该这么早收到业务事件，但万一收到也别丢
                    self._handle_event(msg)
        except queue_mod.Empty:
            pass
        except Exception:
            pass

        # 启动主进程内的 dispatcher 线程
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True, name='VoiceDispatcher')
        self._dispatcher_thread.start()

    def _dispatcher_loop(self):
        """主进程内的轻量线程：从 mp.Queue 拿 'pause'/'resume' 串调用户回调"""
        self._log('>>> [Voice] 派发线程启动')
        while self._running:
            try:
                # mp.Queue 的 get 是支持 timeout 的
                msg = self._event_q.get(timeout=0.5)
            except queue_mod.Empty:
                # 顺便检查子进程是不是挂了
                if self._proc is not None and not self._proc.is_alive():
                    self._log('>>> [Voice] 检测到子进程已退出！')
                    break
                continue
            except Exception as e:
                self._log(f'>>> [Voice] 派发取消息异常: {e}')
                continue
            self._handle_event(msg)
        self._log('>>> [Voice] 派发线程退出')

    def _handle_event(self, msg):
        if isinstance(msg, str) and msg.startswith('error:'):
            self._log(f'>>> [Voice] 子进程错误: {msg[6:]}')
            return
        try:
            if msg == 'pause':
                self.on_pause()
            elif msg == 'resume':
                self.on_resume()
            else:
                self._log(f'>>> [Voice] 未知事件: {msg}')
        except Exception as e:
            self._log(f'>>> [Voice] 回调执行异常: {e}')

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()
        # 等子进程退出
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._log('>>> [Voice] 子进程不肯退出，强制 terminate')
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=2.0)



# ==================================================
class TennisPickNode(Node):
    def __init__(self, weights_path='./best.pt'):
        super().__init__('tennis_pick_node')
        self.get_logger().info('>>> 初始化 tennis_pick_node (Nav2 版)')

        # ---------- ros2 参数：箱子位置（可在命令行覆盖） ----------
        # 用 NaN 作为"未设定"标记
        self.declare_parameter('box_x', float('nan'))
        self.declare_parameter('box_y', float('nan'))
        self.declare_parameter('box_yaw_deg', float('nan'))
        self.declare_parameter('use_initial_pose_as_box', True)
        # 是否启动时自动往 /initialpose 推一次（AMCL 必需，slam_toolbox 无害）
        self.declare_parameter('publish_initial_pose', True)

        # ---------- YOLO ----------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride, self.names = self.model.stride, self.model.names
        self.get_logger().info(f'YOLO loaded on {self.device}')

        # ---------- 图像缓存 ----------
        self.bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_frame = None
        self._latest_detections = []
        self._latest_frame_time = 0.0

        # ---------- TF (SLAM/AMCL 位姿) ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- ToF ----------
        self._tof_lock = threading.Lock()
        self._tof_m = None

        # ---------- Callback Groups ----------
        self.sub_cb_group = ReentrantCallbackGroup()
        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.nav_cb_group = MutuallyExclusiveCallbackGroup()  # Nav2 单独一组

        # ---------- 订阅 ----------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, '/camera/image_color', self._image_cb, qos,
                                 callback_group=self.sub_cb_group)
        self.create_subscription(Range, '/range_0', self._tof_cb, qos,
                                 callback_group=self.sub_cb_group)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # /initialpose 用 transient_local QoS（AMCL 期望的）
        initial_pose_qos = QoSProfile(
    	reliability=QoSReliabilityPolicy.BEST_EFFORT,    
    	history=QoSHistoryPolicy.KEEP_LAST,
    	durability=QoSDurabilityPolicy.VOLATILE,        
    	depth=1,
)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, INITIAL_POSE_TOPIC, initial_pose_qos)

        # ---------- Action Clients ----------
        self.move_client = ActionClient(self, Move, '/move',
                                        callback_group=self.action_cb_group)
        self.arm_client = ActionClient(self, MoveArm, '/move_arm',
                                       callback_group=self.action_cb_group)
        self.gripper_client = ActionClient(self, GripperControl, '/gripper',
                                           callback_group=self.action_cb_group)
        # Nav2
        self.nav2_client = ActionClient(self, NavigateToPose, NAV2_ACTION_NAME,
                                        callback_group=self.nav_cb_group)

        # ---------- 状态 ----------
        self.current_state = 'INIT'
        self.target_bbox = None
        self.detect_count = 0
        self.lost_count = 0
        self.accumulated_search_angle = 0.0
        self.running = True
        self.ctrl_thread = None

        self.box_pose = None  # (x, y, yaw_deg) in map frame

        # ---------- 语音暂停 ----------
        # pause_event.set() 表示"暂停中"；clear() 表示"正常运行"
        self._pause_event = threading.Event()
        # Nav2 当前 goal handle（用于在暂停时取消导航）
        self._current_nav_handle = None
        self._current_nav_lock = threading.Lock()

        # 启动语音线程
        try:
            self.voice = VoiceController(
                on_pause=self._on_voice_pause,
                on_resume=self._on_voice_resume,
                logger=self.get_logger(),
            )
            self.voice.start()
        except Exception as e:
            self.get_logger().error(f'>>> 语音模块初始化失败，继续运行但无语音控制: {e}')
            self.voice = None

        self.get_logger().info('>>> 节点初始化完成')

    # ==================================================
    #  订阅回调
    # ==================================================
    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            detections = self._yolo_detect(frame)
            now = time.time()
            with self._img_lock:
                self._latest_frame = frame
                self._latest_detections = detections
                self._latest_frame_time = now
        except Exception as e:
            self.get_logger().warn(f'image_cb: {e}')

    def _tof_cb(self, msg):
        r = msg.range
        if 0 < r < 8.0:
            with self._tof_lock:
                self._tof_m = r

    def _get_frame_and_detections(self):
        with self._img_lock:
            return (self._latest_frame.copy() if self._latest_frame is not None else None,
                    list(self._latest_detections))

    def _wait_fresh_frame(self, after_time, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._img_lock:
                if self._latest_frame_time > after_time and self._latest_frame is not None:
                    return self._latest_frame.copy(), list(self._latest_detections)
            time.sleep(0.03)
        with self._img_lock:
            return (self._latest_frame.copy() if self._latest_frame is not None else None,
                    list(self._latest_detections))

    def _get_tof_m(self):
        with self._tof_lock:
            return self._tof_m

    # ==================================================
    #  YOLO
    # ==================================================
    def _yolo_detect(self, frame):
        img = letterbox(frame, 640, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)
            pred = non_max_suppression(pred, 0.5, 0.45)
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.names[int(cls)],
                    })
        return detections

    def _draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class_name']}:{det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        pose = self.get_slam_pose()
        pose_str = f'({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.0f})' if pose else 'N/A'
        tof = self._get_tof_m()
        tof_str = f'{tof*100:.1f}cm' if tof is not None else 'N/A'
        cv2.putText(frame, f'STATE: {self.current_state}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'AMCL: {pose_str}  ToF: {tof_str}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        if self.is_paused():
            cv2.putText(frame, '** VOICE PAUSED **', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    # ==================================================
    #  位姿（map -> base_footprint，AMCL 提供）
    # ==================================================
    def get_slam_pose(self, timeout=0.1):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, math.degrees(yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ==================================================
    #  Action 通用工具
    # ==================================================
    def _spin_future(self, future, timeout):
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.05)

    def _send_action_blocking(self, client, goal, timeout=30.0, name='action'):
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
    #  底盘（抓取阶段精确小位移用，原版保留）
    # ==================================================
    def chassis_move(self, x=0.0, y=0.0, theta_deg=0.0,
                     linear_speed=CHASSIS_LINEAR_SPEED,
                     angular_speed_deg=45,
                     timeout=20.0):
        # 暂停期间不允许底盘 action 起步；等恢复
        self._wait_if_paused(log_tag='chassis_move')
        goal = Move.Goal()
        goal.x = float(x)
        goal.y = float(y)
        goal.theta = float(math.radians(theta_deg))
        goal.linear_speed = float(linear_speed)
        goal.angular_speed = float(math.radians(angular_speed_deg))
        return self._send_action_blocking(
            self.move_client, goal, timeout=timeout, name='move')

    # ==================================================
    #  机械臂 / 夹爪
    # ==================================================
    def arm_move(self, x, z, relative=False, timeout=8.0):
        self._wait_if_paused(log_tag='arm_move')
        goal = MoveArm.Goal()
        goal.x = float(x)
        goal.z = float(z)
        goal.relative = bool(relative)
        return self._send_action_blocking(
            self.arm_client, goal, timeout=timeout, name='move_arm')

    def gripper_set(self, target='open', power=0.8, timeout=4.0):
        goal = GripperControl.Goal()
        goal.target_state = (GRIPPER_OPEN if target == 'open'
                             else GRIPPER_CLOSE if target == 'close'
                             else GRIPPER_PAUSE)
        goal.power = float(power)
        return self._send_action_blocking(
            self.gripper_client, goal, timeout=timeout, name='gripper')

    def arm_home(self):
        self.get_logger().info('>>> 机械臂归位')
        self.arm_move(ARM_HOME_X, ARM_HOME_Z, relative=False)

    # ==================================================
    #  Nav2 相关（新增核心功能）
    # ==================================================
    def wait_for_nav2(self, timeout=30.0):
        """等待 Nav2 NavigateToPose action server 上线"""
        self.get_logger().info('>>> 等待 Nav2 NavigateToPose action...')
        ok = self.nav2_client.wait_for_server(timeout_sec=timeout)
        if not ok:
            self.get_logger().error(
                '>>> Nav2 不可达！检查 nav2_bringup 是否启动、'
                '是否加载了正确的 params_file / map')
            return False
        self.get_logger().info('>>> Nav2 已就绪')
        return True

    def publish_initial_pose(self, x, y, yaw_deg, cov_xy=0.25, cov_yaw=0.07):
        """
        往 /initialpose 推一个位姿。
        AMCL 启动时不知道自己在哪，必须通过这个话题告知初始位置；
        否则 AMCL 粒子云撒满全图，定位不收敛。
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = quaternion_from_yaw(math.radians(yaw_deg))
        # 6x6 协方差（行优先）：x, y, z, roll, pitch, yaw
        cov = [0.0] * 36
        cov[0] = cov_xy   # x-x
        cov[7] = cov_xy   # y-y
        cov[35] = cov_yaw # yaw-yaw
        msg.pose.covariance = cov
        self.initial_pose_pub.publish(msg)
        self.get_logger().info(f'>>> 已发布 /initialpose: ({x:.2f}, {y:.2f}, {yaw_deg:.0f}°)')

    def navigate_to_pose_nav2(self, tx, ty, tyaw_deg,
                              timeout=NAV2_TIMEOUT_SEC,
                              max_retry=NAV2_MAX_RETRY):
        """
        用 Nav2 导航到 (tx, ty, tyaw_deg)，map 坐标系。
        失败会自动重试 max_retry 次。
        语音暂停时会主动取消 goal，恢复后由外层循环重新调用本函数。
        """
        for attempt in range(max_retry + 1):
            # 进入循环前先等暂停结束（避免一暂停就 fail）
            self._wait_if_paused(log_tag='nav2 retry-gate')

            self.get_logger().info(
                f'>>> NAV2 导航 -> ({tx:.2f}, {ty:.2f}, {tyaw_deg:.0f}°) '
                f'尝试 {attempt + 1}/{max_retry + 1}')

            if not self.nav2_client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error('>>> NAV2 server 不可达')
                return False

            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = float(tx)
            goal.pose.pose.position.y = float(ty)
            goal.pose.pose.position.z = 0.0
            goal.pose.pose.orientation = quaternion_from_yaw(math.radians(tyaw_deg))

            self._last_nav_log_dist = 999.0
            send_future = self.nav2_client.send_goal_async(
                goal, feedback_callback=self._nav2_feedback_cb)
            self._spin_future(send_future, timeout=5.0)
            if not send_future.done() or send_future.result() is None:
                self.get_logger().warn('>>> NAV2 send_goal 失败')
                continue

            handle = send_future.result()
            if not handle.accepted:
                self.get_logger().warn('>>> NAV2 goal 被拒绝')
                continue

            # 记录 handle，供语音暂停回调取消
            with self._current_nav_lock:
                self._current_nav_handle = handle

            result_future = handle.get_result_async()
            start = time.time()
            paused_during_nav = False
            while not result_future.done():
                if self.is_paused():
                    # 已在 _on_voice_pause 里 cancel 过；这里等到 result 落定
                    paused_during_nav = True
                if time.time() - start > timeout:
                    self.get_logger().warn('>>> NAV2 超时，主动取消')
                    cancel_future = handle.cancel_goal_async()
                    self._spin_future(cancel_future, timeout=3.0)
                    break
                time.sleep(0.1)

            # 清掉 handle
            with self._current_nav_lock:
                self._current_nav_handle = None

            if not result_future.done():
                continue  # 取消后重试

            status = result_future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('>>> NAV2 到达 ✓')
                return True
            else:
                status_name = {
                    GoalStatus.STATUS_ABORTED: 'ABORTED',
                    GoalStatus.STATUS_CANCELED: 'CANCELED',
                }.get(status, f'STATUS={status}')
                if paused_during_nav:
                    self.get_logger().info(
                        f'>>> NAV2 因暂停而 {status_name}，等待恢复后重试')
                    self._wait_if_paused(log_tag='nav2 post-cancel')
                else:
                    self.get_logger().warn(f'>>> NAV2 {status_name}，准备重试')

        self.get_logger().error('>>> NAV2 多次重试均失败')
        return False

    def _nav2_feedback_cb(self, feedback_msg):
        """Nav2 周期性反馈：剩余距离等。仅在距离明显变化时打印，避免刷屏。"""
        try:
            fb = feedback_msg.feedback
            remaining = fb.distance_remaining
            if abs(remaining - self._last_nav_log_dist) > 0.3:
                self.get_logger().info(f'  NAV2 剩余 {remaining:.2f}m')
                self._last_nav_log_dist = remaining
        except Exception:
            pass

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
        """SEARCHING 状态下找不到球时小角度旋转扫描，转满一圈则游走"""
        search_step = 15.0
        self.get_logger().info(f'>>> 旋转 {search_step}° 搜索')
        self.chassis_move(0, 0, search_step)
        self.accumulated_search_angle += search_step
        if self.accumulated_search_angle >= 360.0:
            self.get_logger().info('>>> 提示: 原地旋转一圈未找到网球，开始游走...')
            self._wander()
            self.accumulated_search_angle = 0.0

    def _wander(self):
        """随机游走，打破原地搜索的死角"""
        wander_yaw = random.uniform(-60.0, 60.0)
        wander_dist = random.uniform(0.3, 0.6)
        self.get_logger().info(f'>>> 游走动作: 转向 {wander_yaw:.1f}°, 前进 {wander_dist:.2f}m')
        self.chassis_move(0, 0, wander_yaw, angular_speed_deg=30)
        self.chassis_move(wander_dist, 0, 0, linear_speed=0.2)

    @staticmethod
    def _normalize_angle(a):
        while a > 180: a -= 360
        while a < -180: a += 360
        return a

    # ==================================================
    #  语音暂停相关
    # ==================================================
    def _on_voice_pause(self):
        """语音 '停' 触发"""
        if self._pause_event.is_set():
            return  # 已经在暂停状态，幂等
        self.get_logger().warn('>>> [PAUSE] 语音暂停触发')
        self._pause_event.set()

        # 1) 立即发零速停车（针对 GRABBING 阶段的 cmd_vel 控制）
        try:
            stop_twist = Twist()
            # 多发几次，覆盖可能的丢包
            for _ in range(5):
                self.cmd_vel_pub.publish(stop_twist)
                time.sleep(0.02)
        except Exception as e:
            self.get_logger().warn(f'>>> [PAUSE] 发停车指令异常: {e}')

        # 2) 取消正在执行的 Nav2 导航（PLACING 阶段）
        with self._current_nav_lock:
            handle = self._current_nav_handle
        if handle is not None:
            try:
                self.get_logger().warn('>>> [PAUSE] 取消当前 Nav2 导航')
                handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f'>>> [PAUSE] 取消 Nav2 异常: {e}')

    def _on_voice_resume(self):
        """语音 '开始' 触发"""
        if not self._pause_event.is_set():
            return
        self.get_logger().warn('>>> [RESUME] 语音恢复触发')
        self._pause_event.clear()

    def is_paused(self):
        return self._pause_event.is_set()

    def _wait_if_paused(self, log_tag=''):
        """
        被各种运动函数调用：如果在暂停状态，阻塞等待恢复。
        返回 True 表示曾经暂停过（调用者可借此判断是否要重做某些动作）。
        """
        if not self._pause_event.is_set():
            return False
        if log_tag:
            self.get_logger().info(f'>>> [PAUSE] {log_tag} 等待恢复...')
        # 周期性检查，让 rclpy 不至于假死
        while self._pause_event.is_set() and self.running and rclpy.ok():
            time.sleep(0.1)
        if self.running and rclpy.ok():
            self.get_logger().info('>>> [RESUME] 继续执行')
        return True

    # ==================================================
    #  状态机
    # ==================================================
    def run_searching(self):
        frame, detections = self._get_frame_and_detections()
        if frame is None:
            time.sleep(0.1); return 'SEARCHING'
        high = [d for d in detections
                if d['class_name'] == 'tennis' and d['confidence'] > 0.7]
        if high:
            self.detect_count += 1
            self.lost_count = 0
            if self.detect_count >= DETECT_CONFIRM_FRAMES:
                self.target_bbox = self._select_nearest_ball(high)['bbox']
                self.detect_count = 0
                self.accumulated_search_angle = 0.0
                self.get_logger().info('>>> SEARCHING: 确认目标 -> ROTATING')
                return 'ROTATING'
        else:
            self.lost_count += 1
            self.detect_count = 0
            if self.lost_count >= LOST_TRIGGER_FRAMES:
                self.lost_count = 0
                self._small_angle_search()
        time.sleep(0.1)
        return 'SEARCHING'

    def _flush_image(self):
        with self._img_lock:
            self._latest_frame = None
            self._latest_detections = []

    def run_rotating(self):
        """严格用时间戳保证读到的图像晚于旋转完成时刻"""
        max_retry = 30
        last_offset = None
        oscillation_count = 0
        damping = 0.7
        rotation_done_time = time.time()

        for cnt in range(max_retry):
            frame, detections = self._wait_fresh_frame(rotation_done_time, timeout=1.5)
            if frame is None:
                self.get_logger().warn('>>> ROTATING: 等不到新帧')
                time.sleep(0.2)
                continue

            high = [d for d in detections
                    if d['class_name'] == 'tennis' and d['confidence'] > 0.6]
            if not high:
                self.get_logger().info('>>> ROTATING: 丢失目标 -> SEARCHING')
                return 'SEARCHING'

            nearest = self._select_nearest_ball(high)
            x1, _, x2, _ = nearest['bbox']
            target_center_x = (x1 + x2) / 2
            offset = target_center_x - IMG_CENTER_X

            if abs(offset) <= ROTATE_TOLERANCE_PX:
                self.get_logger().info(f'>>> ROTATING: offset={offset:+.0f} ✓ 对准完成')
                self.target_bbox = nearest['bbox']
                return 'GRABBING'

            if last_offset is not None and last_offset * offset < 0:
                oscillation_count += 1
                damping *= 0.6
                self.get_logger().warn(
                    f'>>> ROTATING: 振荡 #{oscillation_count} damping={damping:.2f}')
            last_offset = offset

            angle = (offset / (IMG_W / 2)) * 50 * damping
            angle = max(-15, min(15, angle))
            if 0 < abs(angle) < 1.5:
                angle = 1.5 if angle > 0 else -1.5

            self.get_logger().info(f'>>> ROTATING: offset={offset:+.0f} -> 旋转 {angle:+.1f}°')
            self.chassis_move(0, 0, -angle, angular_speed_deg=20)

            time.sleep(0.5)
            rotation_done_time = time.time()

            debug_frame, debug_dets = self._wait_fresh_frame(rotation_done_time, timeout=1.0)
            if debug_frame is not None:
                debug_high = [d for d in debug_dets
                              if d['class_name'] == 'tennis' and d['confidence'] > 0.6]
                if debug_high:
                    db = self._select_nearest_ball(debug_high)
                    dcx = (db['bbox'][0] + db['bbox'][2]) / 2
                    new_offset = dcx - IMG_CENTER_X
                    self.get_logger().info(
                        f'   [延迟检测] 旋转前 offset={offset:+.0f}, '
                        f'转 {-angle:+.1f}° 后 offset={new_offset:+.0f}')

        self.get_logger().info('>>> ROTATING: 次数耗尽 -> SEARCHING')
        return 'SEARCHING'

    def run_grabbing(self):
        """前进逼近 + 视觉闭环修正。这部分不走 Nav2，是亚米级视觉伺服。"""
        self.get_logger().info('>>> GRABBING: 打开夹爪')
        self.gripper_set(target='open', power=0.8)

        self.get_logger().info('>>> GRABBING: 前进逼近 (边走边修正)')
        start = time.time()
        arrived = False
        twist = Twist()

        while time.time() - start < 10:
            # 暂停响应：先停车，再等
            if self.is_paused():
                stop = Twist()
                self.cmd_vel_pub.publish(stop)
                self._wait_if_paused(log_tag='GRABBING cmd_vel')
                # 恢复后刷新计时器，避免被吃掉时间预算
                start = time.time()
                continue

            d = self._get_tof_m()
            if d is not None:
                self.get_logger().info(f'  ToF={d*100:.1f}cm')
                if 0 < d <= TOF_TRIGGER_M:
                    arrived = True
                    break

            frame, detections = self._get_frame_and_detections()
            angular_correction = 0.0
            if frame is not None:
                high = [det for det in detections
                        if det['class_name'] == 'tennis' and det['confidence'] > 0.5]
                if high:
                    nearest = self._select_nearest_ball(high)
                    x1, _, x2, _ = nearest['bbox']
                    cx = (x1 + x2) / 2
                    offset = cx - IMG_CENTER_X
                    angular_correction = -offset / 320.0 * 0.3
                    angular_correction = max(-0.5, min(0.5, angular_correction))

            twist.linear.x = 0.18
            twist.angular.z = angular_correction
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.3)

        if not arrived:
            self.get_logger().warn('>>> GRABBING: 未到位 -> HOMING')
            return 'HOMING'

        self.get_logger().info('>>> GRABBING: 闭合夹爪')
        self.gripper_set(target='close', power=0.8)
        time.sleep(0.3)

        self.get_logger().info('>>> GRABBING: 抬起机械臂')
        self.arm_move(ARM_GRAB_LIFT_X, ARM_GRAB_LIFT_Z, relative=False)
        return 'PLACING'

    def run_placing(self):
        """★★★ 用 Nav2 自动导航到箱子（带避障 + 路径规划 + 激光精细对齐）★★★"""
        self.get_logger().info('>>> PLACING: Nav2 导航到箱子')

        if self.box_pose is None:
            self.get_logger().error('>>> PLACING: box_pose 未设定 -> HOMING')
            return 'HOMING'

        bx, by, byaw = self.box_pose
        self.get_logger().info(f'  目标 ({bx:.2f}, {by:.2f}, {byaw:.0f}°)')

        # 调用 Nav2
        ok = self.navigate_to_pose_nav2(bx, by, byaw)
        if not ok:
            self.get_logger().warn('>>> PLACING: Nav2 失败 -> 直接松爪 + HOMING')
            # 即使导航失败也松开夹爪，避免一直夹着球
            self.gripper_set(target='open', power=0.8)
            return 'HOMING'

        self.get_logger().info('>>> PLACING: 松开夹爪放球')
        self.gripper_set(target='open', power=0.8)
        time.sleep(0.5)

        # 后退 + 转身：用原生 Move 而不是 Nav2，因为是固定的小位移
        self.get_logger().info('>>> PLACING: 后退 50cm')
        self.chassis_move(-0.5, 0, 0, linear_speed=0.3)

        self.get_logger().info('>>> PLACING: 旋转 180°')
        self.chassis_move(0, 0, 180, angular_speed_deg=60)

        return 'HOMING'

    def run_homing(self):
        self.get_logger().info('>>> HOMING')
        self.arm_home()
        self.gripper_set(target='open', power=0.8)
        self.detect_count = 0
        self.lost_count = 0
        self.get_logger().info('>>> HOMING: 完成\n')
        return 'SEARCHING'

    # ==================================================
    #  初始化箱子位置
    # ==================================================
    def _init_box_pose(self):
        """
        箱子位置确定优先级：
          1. ros2 参数 box_x / box_y / box_yaw_deg（推荐：标定一次填进去）
          2. 启动时记录的当前 TF 位姿（兜底，等同原版逻辑）
        """
        bx = self.get_parameter('box_x').value
        by = self.get_parameter('box_y').value
        byaw = self.get_parameter('box_yaw_deg').value
        use_initial = self.get_parameter('use_initial_pose_as_box').value

        params_valid = (not math.isnan(bx)) and (not math.isnan(by)) and (not math.isnan(byaw))

        if params_valid:
            self.box_pose = (float(bx), float(by), float(byaw))
            self.get_logger().info(
                f'>>> 用参数指定的箱子位置: ({bx:.2f}, {by:.2f}, {byaw:.0f}°)')
            return

        if use_initial:
            self.get_logger().info('>>> 等待 AMCL 初始化并记录起点为箱子位置...')
            for _ in range(20):
                pose = self.get_slam_pose(timeout=1.0)
                if pose is not None:
                    self.box_pose = pose
                    self.get_logger().info(
                        f'>>> 已记录起点 (BOX): x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.0f}°')
                    return
                time.sleep(0.5)

        self.get_logger().error('>>> 无法获取箱子位置，采用默认 (0,0,0)')
        self.box_pose = (0.0, 0.0, 0.0)

    # ==================================================
    #  循环
    # ==================================================
    def control_loop(self):
        self.get_logger().info('>>> 控制循环启动')
        time.sleep(1.5)

        # 等 Nav2 起来
        if not self.wait_for_nav2(timeout=20.0):
            self.get_logger().warn('>>> Nav2 未就绪，PLACING 阶段将失败！'
                                   '请确认已启动 nav2_bringup')

        self.arm_home()
        self.gripper_set(target='open', power=0.8)

        # 给 AMCL 发初始位姿（让粒子云收敛）
        # 注意：第一次启动时，TF 可能还没有 map -> base_footprint 链路，
        # 这时拿不到位姿；需要先在 RViz 里手动 2D Pose Estimate 一次，
        # 或者直接用参数指定的箱子位置作为 initial pose 候选
        if self.get_parameter('publish_initial_pose').value:
            self._publish_initial_pose_smart()

        # 记录箱子位置（参数优先，否则用当前位置）
        self._init_box_pose()

        self.current_state = 'SEARCHING'

        while self.running and rclpy.ok():
            # 状态机循环顶部检查暂停：在两个状态之间响应"停"
            self._wait_if_paused(log_tag=f'状态={self.current_state}')
            if not (self.running and rclpy.ok()):
                break
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

    def _publish_initial_pose_smart(self):
        """
        智能发布 initial pose：
        - 如果参数 box_x/box_y/box_yaw_deg 都设定了，且 use_initial_pose_as_box=True，
          说明用户希望"启动位置=箱子位置"，那就用箱子位置作为 initial pose。
        - 否则尝试读当前 TF（要求用户已经在 RViz 里点过 2D Pose Estimate）。
        """
        bx = self.get_parameter('box_x').value
        by = self.get_parameter('box_y').value
        byaw = self.get_parameter('box_yaw_deg').value
        params_valid = (not math.isnan(bx)) and (not math.isnan(by)) and (not math.isnan(byaw))

        # 先尝试读当前 TF
        cur = self.get_slam_pose(timeout=2.0)
        if cur is not None:
            self.get_logger().info(f'>>> 检测到现有 TF 位姿: ({cur[0]:.2f}, {cur[1]:.2f}, {cur[2]:.0f}°)')
            # 多发几次，保证 AMCL 收到
            for _ in range(3):
                self.publish_initial_pose(cur[0], cur[1], cur[2])
                time.sleep(0.3)
            return

        # 没有现有 TF，但用户给了箱子位置参数，说明小车此刻就在箱子位置
        if params_valid and self.get_parameter('use_initial_pose_as_box').value:
            self.get_logger().info(
                '>>> 当前无 TF 链路，用参数指定的箱子位置作为 initial pose')
            for _ in range(5):
                self.publish_initial_pose(bx, by, byaw)
                time.sleep(0.5)
            time.sleep(2.0)  # 等 AMCL 收敛
            return

        self.get_logger().warn(
            '>>> 无法自动发布 initial pose！请在 RViz 里手动点 "2D Pose Estimate"')

    def show_loop(self):
        if HEADLESS:
            self.get_logger().info('HEADLESS 模式：不显示画面，按 Ctrl+C 退出')
            try:
                while self.running and rclpy.ok():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.running = False
            return

        try:
            while self.running and rclpy.ok():
                frame, detections = self._get_frame_and_detections()
                if frame is not None:
                    frame = self._draw(frame, detections)
                    cv2.imshow('Tennis Pick Nav2', frame)
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
        # 先停语音，避免它在 ROS 销毁后还在 log
        if getattr(self, 'voice', None) is not None:
            try:
                self.voice.stop()
            except Exception:
                pass
        if self.ctrl_thread:
            self.ctrl_thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = TennisPickNode(weights_path='./best.pt')
    node.start()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.show_loop()
    except KeyboardInterrupt:
        node.get_logger().info('用户中断')
    finally:
        node.shutdown()
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        # rclpy 可能在 KeyboardInterrupt 时已被信号处理器关过一次了
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
