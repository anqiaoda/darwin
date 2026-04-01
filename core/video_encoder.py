"""
视频编码模块 - 多线程异步编码
使用独立线程异步编码帧数据，避免阻塞主控制循环
"""
import cv2
import numpy as np
import threading
from typing import Optional
from queue import Queue, Empty

from config import VideoConfig
from utils.logger import get_logger


class VideoEncoder:
    """视频编码器 - 多线程异步编码"""

    def __init__(self, config: VideoConfig, quality: int = 75):
        self.config = config
        self.quality = quality
        self._logger = get_logger(__name__)
        self._logger.info(f"VideoEncoder初始化开始, quality={quality}")

        # 编码线程
        self._encode_thread = None
        self._stop_event = threading.Event()

        # 编码输入队列 (帧 -> 编码后的字节)
        self._input_queue = Queue(maxsize=2)

        # 编码输出队列 (编码后的字节)
        self._output_queue = Queue(maxsize=5)

        # 启动编码线程
        self._stop_event = threading.Event()
        self._encode_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._encode_thread.start()
        self._logger.info("编码线程已启动")
        self._logger.info(f"线程是否存活: {self._encode_thread.is_alive()}")

    def _encode_loop(self):
        """编码线程主循环"""
        frame_count = 0
        self._logger.info("编码线程进入循环")
        while not self._stop_event.is_set():
            try:
                # 从输入队列获取帧
                frame = self._input_queue.get(timeout=1.0)
                if frame is None:
                    continue

                frame_count += 1

                # 编码帧
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    _, encoded = cv2.imencode('.jpg', frame, encode_param)
                    encoded_bytes = encoded.tobytes()

                    # 放入输出队列
                    try:
                        self._output_queue.put(encoded_bytes, block=False)
                    except:
                        # 输出队列满了，丢弃最旧的
                        try:
                            self._output_queue.get_nowait()
                            self._output_queue.put(encoded_bytes, block=False)
                        except:
                            pass
                except Exception as e:
                    self._logger.error(f"编码失败: {e}", exc_info=True)
            except Empty:
                continue
            except Exception as e:
                self._logger.error(f"编码循环异常: {e}", exc_info=True)
                continue

    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        编码帧（非阻塞，放入队列）

        Args:
            frame: numpy数组图像

        Returns:
            之前的可用编码数据（可能不是当前帧），如果没有则返回None
        """
        # 异步放入输入队列
        try:
            self._input_queue.put(frame, block=False, timeout=0.01)
        except:
            # 队列满了，丢弃当前的帧（不丢弃队列里的，让编码线程继续处理）
            pass

        # 返回最新的可用编码数据（这可能是上一帧编码的结果）
        return self.get_latest_encoded()

    def get_latest_encoded(self) -> Optional[bytes]:
        """
        获取最新的编码数据

        Returns:
            最新的编码数据，如果没有则返回None
        """
        # 取出队列中所有编码数据，只保留最新的
        latest = None
        try:
            while True:
                latest = self._output_queue.get_nowait()
        except Empty:
            pass
        return latest

    def close(self):
        """关闭编码器"""
        self._stop_event.set()
        if self._encode_thread:
            self._encode_thread.join(timeout=1)
        self._logger.info("编码器已关闭")

    def __del__(self):
        """析构函数"""
        self.close()