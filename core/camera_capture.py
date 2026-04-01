"""
深度相机视频流采集模块
实时从深度相机获取视频流数据
使用后台线程持续读帧，避免主循环被阻塞
"""
import cv2
import time
import threading
from typing import Optional, Tuple
from threading import Lock

from config import CameraConfig
from utils.logger import get_logger


class CameraCapture:
    """相机采集器 - 后台线程持续读帧"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = Lock()
        self._logger = get_logger(__name__)
        self._last_frame_time = 0
        self._frame_count = 0

        # 后台读帧线程相关
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_frame: Optional[Tuple[bool, any]] = None
        self._frame_lock = Lock()
        self._new_frame_event = threading.Event()

    def open(self) -> bool:
        """打开相机设备"""
        self._lock.acquire()
        try:
            # 根据配置选择后端
            if self.config.backend == "dshow":
                cap_index = cv2.CAP_DSHOW + self.config.device_id if hasattr(cv2, 'CAP_DSHOW') else self.config.device_id
                self.cap = cv2.VideoCapture(cap_index)
            else:
                self.cap = cv2.VideoCapture(self.config.device_id)

            if not self.cap.isOpened():
                self._logger.error(f"无法打开相机设备 {self.config.device_id}")
                # 尝试其他设备ID
                for test_id in range(5):
                    if test_id == self.config.device_id:
                        continue
                    self.cap = cv2.VideoCapture(test_id)
                    if self.cap.isOpened():
                        self._logger.info(f"成功打开相机设备 {test_id}")
                        self.config.device_id = test_id
                        break
                    else:
                        self._logger.warning(f"尝试设备ID {test_id} 失败")
                if not self.cap.isOpened():
                    return False

            # 设置相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            self._logger.info(
                f"相机已打开: {self.config.width}x{self.config.height} @{self.config.fps}fps"
            )

            # 启动后台读帧线程
            self._stop_event.clear()
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            self._logger.info("后台读帧线程已启动")

            return True

        finally:
            self._lock.release()

    def _read_loop(self):
        """后台线程持续读帧，保持最新帧可用"""
        self._logger.info("读帧线程进入循环")
        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = (ret, frame)
                self._new_frame_event.set()
            else:
                time.sleep(0.001)

        self._logger.info("读帧线程结束")

    def read(self) -> Optional[Tuple[bool, any]]:
        """读取最新一帧图像（非阻塞，返回后台线程最新采集的帧）

        Returns:
            (ret, frame) 元组，如果没有可用帧返回 None
        """
        with self._frame_lock:
            return self._latest_frame

    def read_wait(self, timeout: float = 0.1) -> Optional[Tuple[bool, any]]:
        """等待并读取最新一帧图像

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            (ret, frame) 元组，如果超时返回 None
        """
        self._new_frame_event.clear()
        self._new_frame_event.wait(timeout=timeout)
        with self._frame_lock:
            return self._latest_frame

    def release(self):
        """释放相机资源"""
        self._stop_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)

        self._lock.acquire()
        try:
            if self.cap is not None:
                self.cap.release()
                self._logger.info("相机已释放")
        finally:
            self._lock.release()

    def is_opened(self) -> bool:
        """检查相机是否已打开"""
        return self.cap is not None and self.cap.isOpened()
