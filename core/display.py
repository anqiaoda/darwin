"""
显示模块
实时显示原始和处理后的视频流
"""
import cv2
import time
from typing import Optional

from config import DisplayConfig
from utils.logger import get_logger


class Display:
    """视频显示器"""

    def __init__(self, config: DisplayConfig):
        self.config = config
        self._logger = get_logger(__name__)
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._fps = 0
        self._initialized = False

    def _init_windows(self):
        """初始化可缩放窗口"""
        if self._initialized:
            return

        # 只初始化原始窗口
        cv2.namedWindow(self.config.window_name_original, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_name_original, 800, 600)

        self._initialized = True
        self._logger.info("窗口已初始化，支持拖动和缩放")

    def _prepare_frame(self, frame):
        """准备显示帧（缩放）"""
        if frame is None:
            return None

        # 缩放显示
        display_frame = frame
        if self.config.scale_factor != 1.0:
            h, w = frame.shape[:2]
            display_frame = cv2.resize(
                frame,
                (int(w * self.config.scale_factor),
                 int(h * self.config.scale_factor))
            )

        return display_frame

    def show_original(self, frame):
        """
        显示原始帧

        Args:
            frame: 要显示的原始图像帧
        """
        self._init_windows()
        display_frame = self._prepare_frame(frame)
        if display_frame is not None:
            cv2.imshow(self.config.window_name_original, display_frame)

    def show_processed(self, frame):
        """
        显示处理后的帧

        Args:
            frame: 要显示的处理后图像帧
        """
        self._init_windows()
        display_frame = self._prepare_frame(frame)
        if display_frame is not None:
            cv2.imshow(self.config.window_name_processed, display_frame)

    def wait_key(self, delay: int = 1) -> int:
        """
        等待按键

        Args:
            delay: 等待时间(毫秒)

        Returns:
            按键码
        """
        return cv2.waitKey(delay) & 0xFF

    def destroy(self):
        """销毁显示窗口"""
        cv2.destroyAllWindows()
        self._logger.info("显示窗口已关闭")