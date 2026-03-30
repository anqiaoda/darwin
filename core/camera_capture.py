"""
深度相机视频流采集模块
实时从深度相机获取视频流数据
"""
import cv2
import time
from typing import Optional, Tuple
from threading import Lock

from config import CameraConfig
from utils.logger import get_logger


class CameraCapture:
    """相机采集器"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = Lock()
        self._logger = get_logger(__name__)
        self._last_frame_time = 0
        self._frame_count = 0

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
            return True

        finally:
            self._lock.release()

    def read(self) -> Optional[Tuple[bool, any]]:
        """读取一帧图像"""
        if self.cap is None:
            self._logger.error("相机未打开")
            return None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """释放相机资源"""
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