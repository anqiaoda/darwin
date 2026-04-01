"""
显示模块 - 多线程版本
使用独立线程异步渲染窗口，避免阻塞主控制循环
"""
import cv2
import time
import threading
from typing import Optional
from queue import Queue, Empty

from config import DisplayConfig
from utils.logger import get_logger


class Display:
    """视频显示器 - 多线程异步渲染"""

    def __init__(self, config: DisplayConfig):
        self.config = config
        self._logger = get_logger(__name__)
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._fps = 0
        self._initialized = False

        # 显示线程
        self._display_thread = None
        self._stop_event = threading.Event()

        # 帧队列
        self._original_queue = Queue(maxsize=2)  # 限制队列大小，避免堆积
        self._processed_queue = Queue(maxsize=2)

        # 退出信号
        self._exit_key = None

        # 启动显示线程
        self._start_display_thread()

    def _start_display_thread(self):
        """启动显示线程"""
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        self._logger.info("显示线程已启动")

    def _display_loop(self):
        """显示线程主循环"""
        self._init_windows()

        while not self._stop_event.is_set():
            # 检查退出条件
            if self.config.show_original:
                try:
                    window_closed = (
                        cv2.getWindowProperty(self.config.window_name_original, cv2.WND_PROP_VISIBLE) < 1
                    )
                    if window_closed:
                        self._logger.info("原始窗口已关闭")
                        self._exit_key = 27
                        break
                except:
                    pass

            if self.config.show_processed:
                try:
                    window_closed = (
                        cv2.getWindowProperty(self.config.window_name_processed, cv2.WND_PROP_VISIBLE) < 1
                    )
                    if window_closed:
                        self._logger.info("处理窗口已关闭")
                        self._exit_key = 27
                        break
                except:
                    pass

            # 显示原始帧（每次循环都尝试显示）
            if self.config.show_original:
                try:
                    frame = self._original_queue.get_nowait()
                    display_frame = self._prepare_frame(frame)
                    if display_frame is not None:
                        cv2.imshow(self.config.window_name_original, display_frame)
                except Empty:
                    pass

            # 显示处理帧
            if self.config.show_processed:
                try:
                    frame = self._processed_queue.get_nowait()
                    display_frame = self._prepare_frame(frame)
                    if display_frame is not None:
                        cv2.imshow(self.config.window_name_processed, display_frame)
                except Empty:
                    pass

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self._exit_key = 27
                break
            elif key == ord('q'):
                self._exit_key = ord('q')
                break

            # 短暂休眠，避免CPU占用过高
            time.sleep(0.001)

    def _init_windows(self):
        """初始化可缩放窗口"""
        if self._initialized:
            return

        if self.config.show_original:
            cv2.namedWindow(self.config.window_name_original, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.config.window_name_original, 800, 600)

        if self.config.show_processed:
            cv2.namedWindow(self.config.window_name_processed, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.config.window_name_processed, 800, 600)

        self._initialized = True
        self._logger.info("窗口已初始化，支持拖动和缩放")

    def _prepare_frame(self, frame):
        """准备显示帧（缩放 + FPS绘制）"""
        if frame is None:
            return None

        # 缩放显示
        display_frame = frame.copy()
        if self.config.scale_factor != 1.0:
            h, w = frame.shape[:2]
            display_frame = cv2.resize(
                frame,
                (int(w * self.config.scale_factor),
                 int(h * self.config.scale_factor))
            )

        # 绘制FPS
        if self.config.show_fps:
            self._frame_count += 1
            current_time = time.time()
            if current_time - self._last_frame_time >= 1.0:
                self._fps = self._frame_count / (current_time - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = current_time

            cv2.putText(
                display_frame,
                f"FPS: {self._fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        return display_frame

    def show_original(self, frame):
        """
        显示原始帧（非阻塞，放入队列）

        Args:
            frame: 要显示的原始图像帧
        """
        if not self.config.show_original:
            return

        try:
            self._original_queue.put(frame, block=False)
        except:
            # 队列满了，丢弃旧帧
            try:
                self._original_queue.get_nowait()
                self._original_queue.put(frame, block=False)
            except:
                pass

    def show_processed(self, frame):
        """
        显示处理后的帧（非阻塞，放入队列）

        Args:
            frame: 要显示的处理后图像帧
        """
        if not self.config.show_processed:
            return

        try:
            self._processed_queue.put(frame, block=False)
        except:
            # 队列满了，丢弃旧帧
            try:
                self._processed_queue.get_nowait()
                self._processed_queue.put(frame, block=False)
            except:
                pass

    def wait_key(self, delay: int = 1) -> int:
        """
        等待按键（已废弃，按键由显示线程处理）

        Args:
            delay: 等待时间(毫秒)，已忽略

        Returns:
            退出按键码，如果没有退出返回0
        """
        if self._exit_key:
            return self._exit_key
        return 0

    def check_exit(self) -> bool:
        """
        检查是否收到退出信号

        Returns:
            True表示应该退出，False表示继续运行
        """
        return self._exit_key is not None or self._stop_event.is_set()

    def destroy(self):
        """销毁显示窗口"""
        self._stop_event.set()
        if self._display_thread:
            self._display_thread.join(timeout=1)
        cv2.destroyAllWindows()
        self._logger.info("显示窗口已关闭")