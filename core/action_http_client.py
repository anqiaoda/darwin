"""
动作HTTP客户端模块
通过HTTP调用动作数据服务，获取机器人关节动作数据
"""
import time
import json
import requests
from typing import Optional, Any

from config import HTTPConfig
from utils.logger import get_logger


class ActionHTTPClient:
    """动作数据HTTP客户端"""

    def __init__(self, config: HTTPConfig):
        self.config = config
        self._session = requests.Session()
        self._logger = get_logger(__name__)
        self._last_request_time = 0
        self._request_count = 0

    def send_image(self, image_bytes: bytes) -> Optional[dict]:
        """
        发送图片到动作服务，获取关节动作数据

        根据demo_http/http_action_server.py的接口规范：
        - 输入：图片（POST 请求中的 multipart/form-data，字段名为 'image'）
        - 输出：JSON格式，包含 'time' 和 'q' 字段（关节角度数组）

        Args:
            image_bytes: 图片字节数据

        Returns:
            动作数据字典，包含关节角度数组，失败返回None
        """
        url = f"{self.config.base_url}{self.config.endpoint}"

        try:
            # 使用字段名 'image'（根据http_action_server.py的要求）
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            response = self._session.post(
                url,
                files=files,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                self._request_count += 1

                # 验证返回数据格式
                if 'q' in result:
                    return result
                else:
                    self._logger.error(f"返回数据格式错误: {result}")
                    return None
            else:
                self._logger.warning(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return None

        except requests.exceptions.Timeout:
            self._logger.warning("请求超时")
            return None
        except requests.exceptions.RequestException as e:
            self._logger.error(f"请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON解析失败: {e}")
            return None

    def send_frame(self, frame_bytes: bytes) -> Optional[dict]:
        """
        发送视频帧数据到动作服务，获取关节动作数据

        根据demo_http/http_action_server.py的/process/frame接口规范：
        - 输入：Base64 编码的图像数据（JSON格式）或原始二进制数据
        - 输出：JSON格式，包含 'time' 和 'q' 字段（关节角度数组）

        Args:
            frame_bytes: 帧字节数据（JPEG编码）

        Returns:
            动作数据字典，包含关节角度数组，失败返回None
        """
        # 使用 /process/frame 接口
        url = f"{self.config.base_url}/process/frame"

        try:
            # 发送原始二进制数据，无需图片格式转换
            response = self._session.post(
                url,
                data=frame_bytes,
                timeout=self.config.timeout,
                headers={"Content-Type": "application/octet-stream"}
            )

            if response.status_code == 200:
                result = response.json()
                self._request_count += 1

                # 验证返回数据格式
                if 'q' in result:
                    return result
                else:
                    self._logger.error(f"返回数据格式错误: {result}")
                    return None
            else:
                self._logger.warning(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return None

        except requests.exceptions.Timeout:
            self._logger.warning("请求超时")
            return None
        except requests.exceptions.RequestException as e:
            self._logger.error(f"请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON解析失败: {e}")
            return None

    def close(self):
        """关闭HTTP会话"""
        self._session.close()
        self._logger.info(f"动作HTTP客户端已关闭，共发送 {self._request_count} 个请求")