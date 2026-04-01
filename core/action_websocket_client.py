"""
动作WebSocket客户端模块 - 同步版本
通过WebSocket同步发送帧并等待响应（实时模式）
"""
import asyncio
import websockets
import time
import json
from typing import Optional
from threading import Thread, Event

from config import HTTPConfig
from utils.logger import get_logger


class ActionWebSocketClient:
    """动作WebSocket客户端 - 同步等待响应"""

    def __init__(self, config: HTTPConfig):
        self.config = config
        self._logger = get_logger(__name__)
        self._request_count = 0
        self._connected = False
        self._stop_event = Event()

        # WebSocket 连接
        self._websocket = None
        self._loop = None
        self._thread = None

        # 转换HTTP URL为WebSocket URL（参考 test_ws_client.py）
        ws_url = config.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws_url = f"{ws_url}{config.endpoint}"

        # 启动连接
        self._connect()

    def _connect(self):
        """启动WebSocket连接"""
        self._thread = Thread(target=self._run_websocket, daemon=True)
        self._thread.start()

        # 等待连接
        for _ in range(50):  # 最多等待5秒
            time.sleep(0.1)
            if self._connected:
                return

        self._logger.error(f"连接超时: {self._ws_url}")

    async def _websocket_loop(self):
        """WebSocket主循环"""
        retry_delay = 1.0
        max_retry_delay = 30.0
        self._connected = False

        while not self._stop_event.is_set():
            try:
                self._logger.info(f"正在连接WebSocket: {self._ws_url}")
                async with websockets.connect(self._ws_url, max_size=None) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    retry_delay = 1.0  # 重置重试延迟
                    self._logger.info("WebSocket连接成功")

                    while not self._stop_event.is_set():
                        try:
                            await asyncio.sleep(1)
                        except asyncio.CancelledError:
                            break

            except (websockets.ConnectionClosed,
                   websockets.WebSocketException,
                   OSError, ConnectionError) as e:
                self._connected = False
                self._websocket = None
                self._logger.warning(f"连接断开: {e}")

                if not self._stop_event.is_set():
                    self._logger.info(f"{retry_delay:.1f}秒后重连...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)

            except Exception as e:
                self._logger.error(f"WebSocket异常: {e}", exc_info=True)
                self._connected = False
                if not self._stop_event.is_set():
                    await asyncio.sleep(retry_delay)

    def _run_websocket(self):
        """运行WebSocket事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._websocket_loop())
        finally:
            self._connected = False

    async def _send_and_recv(self, frame_bytes: bytes) -> Optional[dict]:
        """异步发送并接收响应"""
        if not self._websocket:
            self._logger.error("WebSocket 未连接")
            return None

        try:
            t0 = time.time()
            await self._websocket.send(frame_bytes)
            self._logger.debug(f"发送成功，等待响应...")

            resp = await self._websocket.recv()
            elapsed = time.time() - t0
            self._logger.debug(f"收到响应，耗时: {elapsed:.3f}s")

            self._request_count += 1

            # 解析响应
            try:
                return json.loads(resp)
            except json.JSONDecodeError:
                self._logger.warning(f"无法解析JSON: {resp[:100]}")
                return None

        except Exception as e:
            self._logger.error(f"发送/接收失败: {e}")
            return None

    def send_frame(self, frame_bytes: bytes) -> Optional[dict]:
        """
        同步发送帧并等待响应（实时模式）
        参考 test_ws_client.py: await ws.send(buf.tobytes()); resp = await ws.recv()

        Args:
            frame_bytes: 帧字节数据（JPEG编码）

        Returns:
            动作数据字典，失败返回None
        """
        if not self._connected or not self._websocket or not self._loop:
            return None

        # 在事件循环中执行异步操作
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_and_recv(frame_bytes),
                self._loop
            )
            # 等待响应（带超时）
            result = future.result(timeout=self.config.timeout)
            return result
        except asyncio.TimeoutError:
            self._logger.warning("请求超时")
            return None
        except Exception as e:
            self._logger.error(f"请求异常: {e}")
            return None

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected and self._websocket is not None

    def close(self):
        """关闭WebSocket连接"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._logger.info(f"WebSocket客户端已关闭，共发送 {self._request_count} 个请求")