"""
动作WebSocket客户端模块 - 流水线模式
发送和接收分离：发送不阻塞，后台线程持续接收结果
"""
import asyncio
import websockets
import time
import json
from typing import Optional
from threading import Thread, Event
from queue import Queue, Empty

from config import HTTPConfig
from utils.logger import get_logger


class ActionWebSocketClient:
    """动作WebSocket客户端 - 流水线模式（发送/接收分离）"""

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

        # 流水线模式
        self._pipeline_started = False
        self._result_queue: Queue = Queue(maxsize=2)

        # 转换HTTP URL为WebSocket URL
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
                    retry_delay = 1.0
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

    # ========== 流水线模式 ==========

    def start_pipeline(self):
        """启动流水线接收线程"""
        if self._pipeline_started:
            return
        self._pipeline_started = True
        self._logger.info("启动流水线接收线程")

        def _recv_loop():
            """后台持续接收 WebSocket 响应，放入结果队列"""
            while not self._stop_event.is_set() and self._connected:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._websocket.recv(),
                        self._loop
                    )
                    resp = future.result(timeout=self.config.timeout)

                    # 解析 JSON
                    try:
                        result = json.loads(resp)
                        self._request_count += 1
                        # 放入队列，满了则丢弃最旧的
                        try:
                            self._result_queue.put(result, block=False)
                        except:
                            try:
                                self._result_queue.get_nowait()
                                self._result_queue.put(result, block=False)
                            except:
                                pass
                    except (json.JSONDecodeError, TypeError):
                        self._logger.debug(f"无法解析JSON响应")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self._logger.debug(f"接收异常: {e}")
                    time.sleep(0.01)

            self._logger.info("流水线接收线程结束")

        t = Thread(target=_recv_loop, daemon=True)
        t.start()

    def send_frame_async(self, frame_bytes: bytes) -> bool:
        """流水线发送：只发送不等待响应

        Args:
            frame_bytes: 帧字节数据（JPEG编码）

        Returns:
            True 发送成功，False 失败
        """
        if not self._connected or not self._websocket or not self._loop:
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._websocket.send(frame_bytes),
                self._loop
            )
            future.result(timeout=self.config.timeout)
            return True
        except Exception as e:
            self._logger.error(f"发送失败: {e}")
            return False

    def get_latest_result(self) -> Optional[dict]:
        """非阻塞获取最新的动作结果（清空旧结果，只保留最新）

        Returns:
            最新的动作数据，无数据返回 None
        """
        latest = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except Empty:
            pass
        return latest

    # ========== 兼容旧模式 ==========

    async def _send_and_recv(self, frame_bytes: bytes) -> Optional[dict]:
        """异步发送并接收响应（旧同步模式）"""
        if not self._websocket:
            self._logger.error("WebSocket 未连接")
            return None

        try:
            await self._websocket.send(frame_bytes)
            resp = await self._websocket.recv()
            self._request_count += 1

            try:
                return json.loads(resp)
            except json.JSONDecodeError:
                self._logger.warning(f"无法解析JSON: resp[:100]")
                return None

        except Exception as e:
            self._logger.error(f"发送/接收失败: {e}")
            return None

    def send_frame(self, frame_bytes: bytes) -> Optional[dict]:
        """同步发送帧并等待响应（旧模式，向后兼容）"""
        if not self._connected or not self._websocket or not self._loop:
            return None

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_and_recv(frame_bytes),
                self._loop
            )
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
