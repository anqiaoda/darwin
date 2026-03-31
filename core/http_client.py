"""
HTTP客户端模块
通过HTTP调用外部模型服务
"""
import json
import time
from typing import Optional, Any
import requests

from config import HTTPConfig
from utils.logger import get_logger


class HTTPClient:
    """HTTP客户端"""

    def __init__(self, config: HTTPConfig):
        self.config = config
        self._session = requests.Session()
        self._logger = get_logger(__name__)
        self._rate_limited_until = 0
        self._consecutive_429 = 0
        self._retry_delay_base = 1.0  # 基础重试延迟

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

    def close(self):
        """关闭HTTP会话"""
        self._session.close()