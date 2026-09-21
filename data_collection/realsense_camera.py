"""
realsense_camera.py

Intel RealSense 相机接口。
采集 RGB 图像用于 OpenVLA 数据集。
使用后台线程持续抓帧，主线程随时取最新帧，避免与 CAN 通信竞争时序。
"""

import threading
import time

import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


class RealSenseCamera:
    """Intel RealSense RGB 相机封装（后台线程持续抓帧）"""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, serial: str = ""):
        if not HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 未安装。请运行: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial

        self.pipeline = None
        self.config = None
        self._started = False

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None   # 未被 capture_rgb 消费的新帧
        self._last_frame: np.ndarray | None = None     # 最近一帧 (仅供预览 peek, 不消费)
        self._grab_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _grab_loop(self):
        """后台线程：持续从 pipeline 取帧，保留最新一帧。"""
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    img = np.asanyarray(color_frame.get_data()).copy()
                    with self._lock:
                        self._latest_frame = img
                        self._last_frame = img
            except RuntimeError:
                pass

    def connect(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)

        self.config.enable_stream(
            rs.stream.color,
            self.width, self.height,
            rs.format.rgb8,
            self.fps,
        )

        self.pipeline.start(self.config)
        self._started = True

        # 丢弃前几帧 (自动曝光稳定)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # 启动后台抓帧线程
        self._stop_event.clear()
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

        print(f"[RealSense] 已连接 ({self.width}x{self.height} @ {self.fps}fps)")

    def disconnect(self):
        if self._grab_thread is not None:
            self._stop_event.set()
            self._grab_thread.join(timeout=3)
            self._grab_thread = None

        if self._started and self.pipeline is not None:
            self.pipeline.stop()
            self._started = False
            print("[RealSense] 已断开")

    def capture_rgb(self, timeout: float = 5.0) -> np.ndarray:
        """
        获取最新一帧 RGB 图像（由后台线程持续更新）。

        Returns:
            np.ndarray: shape (H, W, 3), dtype uint8, RGB 格式
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                    return frame
            time.sleep(0.005)
        raise RuntimeError(f"[RealSense] {timeout}s 内未获取到新帧")

    def peek_rgb(self) -> np.ndarray | None:
        """
        非消费地读取最近一帧 (供实时预览用), 不影响 capture_rgb 的"新帧"语义。
        尚无帧时返回 None。
        """
        with self._lock:
            return self._last_frame

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


class DummyCamera:
    """
    模拟相机，用于没有 RealSense 时的调试。
    返回随机噪声图像。
    """

    def __init__(self, width: int = 640, height: int = 480, **kwargs):
        self.width = width
        self.height = height

    def connect(self):
        print(f"[DummyCamera] 已连接 ({self.width}x{self.height})")

    def disconnect(self):
        print("[DummyCamera] 已断开")

    def capture_rgb(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def peek_rgb(self) -> np.ndarray:
        return self.capture_rgb()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
