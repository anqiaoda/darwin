"""
HTTP客户端模块
通过HTTP调用外部模型服务
支持流水线模式：后台线程发送请求+接收结果，主线程非阻塞获取最新结果
"""
import json
import time
import threading
from typing import Optional, Any
from queue import Queue, Empty
import requests

from config import HTTPConfig
from utils.logger import get_logger


class HTTPClient:
    """HTTP客户端（支持流水线模式）"""

    def __init__(self, config: HTTPConfig):
        self.config = config
        self._session = requests.Session()
        self._logger = get_logger(__name__)
        self._rate_limited_until = 0
        self._consecutive_429 = 0
        self._retry_delay_base = 1.0

        # 流水线模式
        self._pipeline_started = False
        self._pipeline_stop = threading.Event()
        self._send_queue: Queue = Queue(maxsize=2)  # 待发送的帧
        self._result_queue: Queue = Queue(maxsize=2)  # 处理结果

    def _is_rate_limited(self) -> bool:
        """检查是否处于限流状态"""
        return time.time() < self._rate_limited_until

    def _handle_rate_limit(self, response_status: int):
        """处理限流响应"""
        if response_status == 429:
            self._consecutive_429 += 1
            # 指数退避：1s, 2s, 4s, 8s, 16s, 32s, 最大60s
            delay = min(self._retry_delay_base * (2 ** (self._consecutive_429 - 1)), 60)
            self._rate_limited_until = time.time() + delay

            self._logger.warning(
                f"触发限流(429)，等待 {delay:.1f}秒后再试 "
                f"(连续第{self._consecutive_429}次)"
            )
        else:
            self._consecutive_429 = 0

    def send_frame(self, frame_data: bytes, height: int = 480, width: int = 640, channels: int = 3) -> Optional[bytes]:
        """
        发送帧数据到模型服务进行骨骼检测

        Args:
            frame_data: 原始图像字节数据（numpy uint8 数组的字节流）
            height: 图像高度
            width: 图像宽度
            channels: 图像通道数

        Returns:
            模型返回的处理后图片数据 (带骨骼绘制)
        """
        # 检查是否处于限流状态
        if self._is_rate_limited():
            return None

        url = f"{self.config.base_url}{self.config.endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                # 构造数据：前12字节为尺寸信息，后面是图像数据
                header = height.to_bytes(4, byteorder='little') + \
                         width.to_bytes(4, byteorder='little') + \
                         channels.to_bytes(4, byteorder='little')
                payload = header + frame_data

                # 使用原始二进制格式发送
                response = self._session.post(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    self._consecutive_429 = 0  # 成功后重置计数器
                    # API直接返回图片流，不是JSON
                    return response.content
                else:
                    self._handle_rate_limit(response.status_code)
                    self._logger.warning(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    # 如果是限流错误，直接退出重试循环
                    if response.status_code == 429:
                        break

            except requests.exceptions.Timeout:
                self._logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.config.max_retries})")
            except requests.exceptions.RequestException as e:
                self._logger.error(f"请求失败: {e}")

        return None

    def detect_person(self, frame_data: bytes, height: int = 480, width: int = 640, channels: int = 3) -> Optional[dict]:
        """
        检测图像中是否有人体

        Args:
            frame_data: 原始图像字节数据（numpy uint8 数组的字节流）
            height: 图像高度
            width: 图像宽度
            channels: 图像通道数

        Returns:
            检测结果字典，包含 has_person、has_complete_person 等字段；失败返回 None
        """
        url = f"{self.config.base_url}{self.config.endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                # 构造数据：前12字节为尺寸信息，后面是图像数据
                header = height.to_bytes(4, byteorder='little') + \
                         width.to_bytes(4, byteorder='little') + \
                         channels.to_bytes(4, byteorder='little')
                payload = header + frame_data

                response = self._session.post(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    # 检测接口返回 JSON
                    return response.json()
                else:
                    self._logger.warning(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )

            except requests.exceptions.Timeout:
                self._logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.config.max_retries})")
            except requests.exceptions.RequestException as e:
                self._logger.error(f"请求失败: {e}")
            except Exception as e:
                self._logger.error(f"解析响应失败: {e}")

        return None

    # ========== 流水线模式 ==========

    def start_pipeline(self):
        """启动流水线后台线程（发送+接收）"""
        if self._pipeline_started:
            return
        self._pipeline_started = True
        self._logger.info("启动HTTP流水线线程")

        def _pipeline_loop():
            while not self._pipeline_stop.is_set():
                try:
                    # 取待发送的帧
                    item = self._send_queue.get(timeout=0.5)
                    frame_data, height, width, channels = item

                    # 发送请求并获取结果
                    result = self.send_frame(frame_data, height, width, channels)
                    if result is not None:
                        # 放入结果队列，满了丢弃最旧的
                        try:
                            self._result_queue.put(result, block=False)
                        except:
                            try:
                                self._result_queue.get_nowait()
                                self._result_queue.put(result, block=False)
                            except:
                                pass
                except Empty:
                    continue
                except Exception as e:
                    self._logger.debug(f"流水线异常: {e}")

        t = threading.Thread(target=_pipeline_loop, daemon=True)
        t.start()

    def send_frame_async(self, frame_data: bytes, height: int = 480, width: int = 640, channels: int = 3) -> bool:
        """流水线发送：放入发送队列，不等待响应

        Returns:
            True 成功放入队列，False 队列满
        """
        try:
            # 队列满了丢弃最旧的
            if self._send_queue.full():
                try:
                    self._send_queue.get_nowait()
                except:
                    pass
            self._send_queue.put((frame_data, height, width, channels), block=False)
            return True
        except:
            return False

    def get_latest_result(self) -> Optional[bytes]:
        """非阻塞获取最新的处理结果（清空旧结果，只保留最新）"""
        latest = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except Empty:
            pass
        return latest

    # ========== 通用方法 ==========

    def close(self):
        """关闭HTTP会话"""
        self._pipeline_stop.set()
        self._session.close()