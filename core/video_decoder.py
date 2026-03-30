"""
视频解码/编码模块
处理视频流的解码和编码
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from config import VideoConfig
from utils.logger import get_logger


class VideoDecoder:
    """视频解码器"""

    def __init__(self, config: VideoConfig):
        self.config = config
        self._logger = get_logger(__name__)

    def decode_frame(self, encoded_data: bytes) -> Optional[np.ndarray]:
        """
        解码编码后的帧数据

        Args:
            encoded_data: 编码的字节数据

        Returns:
            解码后的numpy数组图像
        """
        try:
            nparr = np.frombuffer(encoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            self._logger.error(f"解码失败: {e}")
            return None

    def encode_frame(self, frame: np.ndarray, quality: int = 90) -> Optional[bytes]:
        """
        编码帧数据

        Args:
            frame: numpy数组图像
            quality: JPEG质量(1-100)

        Returns:
            编码后的字节数据
        """
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            return encoded.tobytes()
        except Exception as e:
            self._logger.error(f"编码失败: {e}")
            return None

    def resize_frame(
        self,
        frame: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        调整帧大小

        Args:
            frame: 原始帧
            width: 目标宽度
            height: 目标高度

        Returns:
            调整大小后的帧
        """
        if width is None:
            width = self.config.output_width
        if height is None:
            height = self.config.output_height

        if width is None and height is None:
            return frame

        h, w = frame.shape[:2]
        if width is None:
            ratio = height / h
            width = int(w * ratio)
        elif height is None:
            ratio = width / w
            height = int(h * ratio)

        return cv2.resize(frame, (width, height))

    def decode_result(self, result: Optional[bytes]) -> Optional[np.ndarray]:
        """
        解码模型返回的结果

        Args:
            result: 模型返回的图片字节数据

        Returns:
            解码后的图像帧
        """
        if result is None:
            return None

        try:
            # API直接返回图片流，直接解码
            return self.decode_frame(result)
        except Exception as e:
            self._logger.error(f"解码结果失败: {e}")
            return None