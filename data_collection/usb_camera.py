"""
usb_camera.py

USB UVC 相机接口 (基于 OpenCV)。
接口与 RealSenseCamera 对齐: connect / disconnect / capture_rgb。
使用后台线程持续抓帧, 主线程随时取最新帧, 避免与 CAN 通信争抢时序。

返回的 RGB 图像格式与 RealSenseCamera 一致 (H, W, 3) uint8 RGB。
"""

import threading
import time
from typing import Optional, Union

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class USBCamera:
    """USB UVC RGB 相机封装 (OpenCV 后台抓帧线程)"""

    def __init__(
        self,
        device: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
    ):
        """
        Args:
            device: cv2.VideoCapture 设备索引 (int, 如 0/2/4) 或路径 (str, 如 "/dev/video2")
            width, height, fps: 期望分辨率/帧率, 实际值取决于相机硬件能力
            fourcc: 像素格式四字符码, 大多数 USB 相机用 "MJPG" 才能跑到 30fps
                   传空串 "" 跳过设置 (用驱动默认值)
        """
        if not HAS_CV2:
            raise ImportError(
                "opencv-python 未安装。请运行: pip install opencv-python"
            )

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

        self.cap: Optional["cv2.VideoCapture"] = None
        self._started = False

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None   # 未被 capture_rgb 消费的新帧
        self._last_frame: Optional[np.ndarray] = None     # 最近一帧 (仅供预览 peek, 不消费)
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _grab_loop(self):
        """后台线程: 持续从 VideoCapture 取帧, 保留最新一帧 (RGB)。"""
        while not self._stop_event.is_set():
            if self.cap is None:
                break
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_frame = rgb
                self._last_frame = rgb

    def connect(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[USBCamera] 无法打开设备 {self.device!r}。"
                f"请确认 USB 相机已插好, 或换一个索引/路径 (e.g. 0, 2, /dev/video2)"
            )

        # 顺序很关键: 先设 fourcc, 再设分辨率, 否则有些驱动会忽略
        if self.fourcc:
            fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 关键: 把内部缓冲设到 1, 否则 cap.read() 会返回积压的旧帧
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps

        self._started = True

        # 预热, 丢弃前几帧 (自动曝光稳定)
        for _ in range(5):
            self.cap.read()

        self._stop_event.clear()
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

        print(
            f"[USBCamera] 已连接 device={self.device!r} "
            f"({actual_w}x{actual_h} @ {actual_fps:.0f}fps)"
        )

    def disconnect(self):
        if self._grab_thread is not None:
            self._stop_event.set()
            self._grab_thread.join(timeout=3)
            self._grab_thread = None

        if self._started and self.cap is not None:
            self.cap.release()
            self.cap = None
            self._started = False
            print("[USBCamera] 已断开")

    def capture_rgb(self, timeout: float = 5.0) -> np.ndarray:
        """
        获取最新一帧 RGB 图像 (由后台线程持续更新)。

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
        raise RuntimeError(f"[USBCamera] {timeout}s 内未获取到新帧")

    def peek_rgb(self) -> Optional[np.ndarray]:
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
