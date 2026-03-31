"""
动作WebSocket客户端模块
通过WebSocket长连接实时获取机器人关节动作数据
（参考 test_ws_client.py 的实现）
"""
import asyncio
import websockets
import time
import json
from typing import Optional, Callable
from threading import Thread, Lock, Event
import queue

from config import HTTPConfig
from utils.logger import get_logger


class ActionWebSocketClient:
    """动作WebSocket客户端"""

    def __init__(self, config: HTTPConfig):
        self.config = config
        self._logger = get_logger(__name__)
        self._request_count = 0
        self._lock = Lock()
        self._connected = False
        self._stop_event = Event()
        self._response_queue = queue.Queue(maxsize=10)  # 增大队列
        self._last_action = None  # 缓存最新的动作数据
        self._last_action_time = 0

        # WebSocket 连接
        self._websocket = None
        self._loop = None
        self._thread = None

        # 响应回调
        self._response_callback: Optional[Callable] = None

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
        for _ in range(50):  # 最多等待2.5秒
            time.sleep(0.05)
            if self._connected:
                return

        self._logger.error(f"连接超时: {self._ws_url}")

    async def _websocket_loop(self):
        """WebSocket主循环"""
        retry_delay = 1.0
        max_retry_delay = 30.0

        while not self._stop_event.is_set():
            try:
                self._logger.info(f"正在连接WebSocket: {self._ws_url}")
                async with websockets.connect(self._ws_url, max_size=None) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    retry_delay = 1.0  # 重置重试延迟
                    self._logger.info("WebSocket连接成功")

                    # 启动接收任务
                    receive_task = asyncio.create_task(self._receive_messages())

                    # 保持连接
                    while not self._stop_event.is_set():
                        try:
                            await asyncio.sleep(1)
                        except asyncio.CancelledError:
                            break

                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

            except (websockets.ConnectionClosed,
                   websockets.WebSocketException,
                   OSError, ConnectionError) as e:
                self._connected = False
                self._logger.warning(f"连接断开: {e}")

                if not self._stop_event.is_set():
                    self._logger.info(f"{retry_delay:.1f}秒后重连...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)

            except Exception as e:
                self._logger.error(f"WebSocket异常: {e}", exc_info=True)
                if not self._stop_event.is_set():
                    await asyncio.sleep(retry_delay)

    async def _receive_messages(self):
        """接收消息循环"""
        try:
            async for message in self._websocket:
                try:
                    # 二进制消息（JPEG图片）- 发送的是图片数据
                    if isinstance(message, bytes):
                        # 忽略，我们只关心动作响应
                        continue

                    # 文本消息（JSON响应）- 接收到的是JSON格式的动作数据
                    elif isinstance(message, str):
                        try:
                            response = json.loads(message)
                            with self._lock:
                                self._request_count += 1
                                self._last_action = response
                                self._last_action_time = time.time()

                            # 有合法响应就缓存
                            if 'motions' in response or 'q' in response:
                                # 打印响应keys和关节数据（调试用）
                                if self._request_count <= 5 or self._request_count % 50 == 0:
                                    print(f'Frame #{self._request_count} - response keys: {list(response.keys())}')
                                    # 打印关节数据
                                    if 'motions' in response and 'dof_pos' in response['motions']:
                                        dof_pos = response['motions']['dof_pos']
                                        print(f'  dof_pos length: {len(dof_pos)}')
                                        print(f'  dof_pos: {dof_pos}')
                                    elif 'q' in response:
                                        print(f'  q length: {len(response["q"])}')
                                    # 打印人数信息
                                    if 'num_people' in response:
                                        print(f'  num_people: {response["num_people"]}')

                                # 放入队列（非阻塞）
                                try:
                                    self._response_queue.put_nowait(response)
                                except queue.Full:
                                    # 队列满了，丢弃最旧的
                                    try:
                                        self._response_queue.get_nowait()
                                        self._response_queue.put_nowait(response)
                                    except queue.Empty:
                                        pass

                                # 调用回调
                                if self._response_callback:
                                    self._response_callback(response)
                            else:
                                # 处理错误响应
                                if 'error' in response:
                                    # 服务端错误：通常是因为没有检测到人体
                                    self._logger.debug(f"未检测到关节数据: {response.get('error', '未知原因')}")
                                else:
                                    # 未知响应格式
                                    self._logger.warning(f"未知响应格式: {list(response.keys())}")
                        except json.JSONDecodeError:
                            self._logger.warning(f"无法解析JSON: {message[:100]}")

                except Exception as e:
                    self._logger.error(f"处理消息失败: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"接收消息异常: {e}")

    def _run_websocket(self):
        """运行WebSocket事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._websocket_loop())
        finally:
            self._connected = False

    def send_frame(self, frame_bytes: bytes) -> Optional[dict]:
        """
        发送帧数据（非阻塞立即返回），使用缓存的最新动作
        参考 test_ws_client.py: await ws.send(buf.tobytes())

        Args:
            frame_bytes: 帧字节数据（JPEG编码）

        Returns:
            最新的动作数据字典（可能为None）
        """
        if not self._connected or not self._websocket:
            return None

        async def _send():
            try:
                await self._websocket.send(frame_bytes)
            except Exception as e:
                self._logger.error(f"发送失败: {e}")
                return False
            return True

        # 发送数据（非阻塞）
        if self._loop and self._websocket:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)

        # 立即返回最新缓存的动作，不等待
        with self._lock:
            return self._last_action

    def send_frame_async(self, frame_bytes: bytes, callback: Optional[Callable] = None):
        """
        异步发送帧，通过回调返回结果
        参考 test_ws_client.py 的异步模式

        Args:
            frame_bytes: 帧字节数据（JPEG编码）
            callback: 响应回调函数
        """
        if not self._connected or not self._websocket:
            if callback:
                callback(None)
            return

        # 设置回调
        if callback:
            self._response_callback = callback

        async def _send():
            try:
                await self._websocket.send(frame_bytes)
            except Exception as e:
                self._logger.error(f"发送失败: {e}")
                if callback:
                    callback(None)

        if self._loop:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def set_response_callback(self, callback: Callable):
        """设置响应回调函数

        Args:
            callback: 回调函数，参数为response或None
        """
        self._response_callback = callback

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "request_count": self._request_count,
                "connected": self._connected,
                "queue_size": self._response_queue.qsize()
            }

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def close(self):
        """关闭WebSocket连接"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._logger.info(f"WebSocket客户端已关闭，共接收 {self._request_count} 个响应")