"""
显示模块 - 高帧率版本
使用双缓冲+独立显示线程，最大化视频窗口帧率
显示线程以最快速度渲染最新帧，不等待主循环
"""
import cv2
import time
import threading
from typing import Optional
from queue import Queue, Empty

from config import DisplayConfig
from utils.logger import get_logger


class Display:
    """视频显示器 - 高帧率双缓冲渲染"""

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

        # 双缓冲：写缓冲（主线程写）和读缓冲（显示线程读）
        self._original_buffer_lock = threading.Lock()
        self._original_write = None  # 主线程最新写入的帧
        self._original_read = None   # 显示线程当前渲染的帧
        self._original_new = threading.Event()  # 有新帧信号

        self._processed_buffer_lock = threading.Lock()
        self._processed_write = None
        self._processed_read = None
        self._processed_new = threading.Event()

        # 退出信号
        self._exit_key = None

        # 启动显示线程
        self._start_display_thread()

    def _start_display_thread(self):
        """启动显示线程"""
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        self._logger.info("显示线程已启动（高帧率模式）")

    def _display_loop(self):
        """显示线程主循环 - 以最快速度渲染"""
        self._init_windows()

        # 无新帧时的轮询间隔
        poll_interval = 0.001  # 1ms

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

            # 交换双缓冲：将写缓冲的内容交换到读缓冲
            has_update = False

            if self.config.show_original:
                with self._original_buffer_lock:
                    if self._original_write is not None:
                        self._original_read = self._original_write
                        self._original_write = None
                        has_update = True

                if self._original_read is not None:
                    display_frame = self._prepare_frame(self._original_read)
                    if display_frame is not None:
                        cv2.imshow(self.config.window_name_original, display_frame)

            if self.config.show_processed:
                with self._processed_buffer_lock:
                    if self._processed_write is not None:
                        self._processed_read = self._processed_write
                        self._processed_write = None
                        has_update = True

                if self._processed_read is not None:
                    display_frame = self._prepare_frame(self._processed_read)
                    if display_frame is not None:
                        cv2.imshow(self.config.window_name_processed, display_frame)

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self._exit_key = 27
                break
            elif key == ord('q'):
                self._exit_key = ord('q')
                break

            # 如果没有更新，短暂休眠避免CPU空转
            if not has_update:
                time.sleep(poll_interval)

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
        display_frame = frame
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

            # 只在需要缩放时才copy，否则直接在原帧上绘制（避免不必要的copy）
            if display_frame is frame:
                display_frame = frame.copy()

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
        显示原始帧（非阻塞，双缓冲写入）

        Args:
            frame: 要显示的原始图像帧
        """
        if not self.config.show_original:
            return

        with self._original_buffer_lock:
            self._original_write = frame

    def show_processed(self, frame):
        """
        显示处理后的帧（非阻塞，双缓冲写入）

        Args:
            frame: 要显示的处理后图像帧
        """
        if not self.config.show_processed:
            return

        with self._processed_buffer_lock:
            self._processed_write = frame

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
