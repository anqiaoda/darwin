"""
Darwin 视频流处理主程序
实时采集深度相机视频流->解码->HTTP调用模型->显示结果
"""
import cv2
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from core.camera_capture import CameraCapture
from core.video_decoder import VideoDecoder
from core.http_client import HTTPClient
from core.display import Display
from utils.logger import get_logger


class DarwinApp:
    """Darwin应用主类"""

    def __init__(self, config: Config):
        self.config = config
        self._logger = get_logger(__name__, config.log_level)

        # 初始化各模块
        self.camera = CameraCapture(config.camera)
        self.decoder = VideoDecoder(config.video)
        self.http_client = HTTPClient(config.http)
        self.display = Display(config.display)

    def run(self):
        """运行主循环"""
        self._logger.info("Darwin应用启动中...")

        # 打开相机
        if not self.camera.open():
            self._logger.error("无法打开相机，程序退出")
            return

        self._logger.info("开始采集和处理视频流...")

        try:
            while True:
                # 1. 从相机读取帧
                data = self.camera.read()
                if data is None:
                    self._logger.error("读取相机失败")
                    break

                ret, frame = data
                if not ret or frame is None:
                    self._logger.error("无法获取有效帧")
                    break

                # 2. 编码帧数据用于传输
                encoded_frame = self.decoder.encode_frame(frame)
                if encoded_frame is None:
                    self._logger.error("帧编码失败")
                    continue

                # 3. 发送到HTTP模型服务处理
                result = self.http_client.send_frame(encoded_frame)

                # 4. 解码处理结果并显示
                display_frame = self.decoder.decode_result(result)
                if display_frame is None:
                    # 如果模型服务失败，显示原始帧
                    display_frame = frame

                # 5. 显示两路视频流（原始+处理后）
                self.display.show_original(frame)
                self.display.show_processed(display_frame)

                # 6. 检查退出按键 (ESC或关闭窗口)
                key = self.display.wait_key(1)
                window_closed = (
                    cv2.getWindowProperty(self.config.display.window_name_original, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(self.config.display.window_name_processed, cv2.WND_PROP_VISIBLE) < 1
                )
                if key == 27 or window_closed:
                    self._logger.info("用户退出程序")
                    break

        except KeyboardInterrupt:
            self._logger.info("收到中断信号，退出程序")

        except Exception as e:
            self._logger.error(f"程序异常: {e}", exc_info=True)

        finally:
            self._cleanup()

    def _cleanup(self):
        """清理资源"""
        self._logger.info("正在清理资源...")
        self.camera.release()
        self.display.destroy()
        self.http_client.close()
        self._logger.info("程序已退出")


def main():
    """程序入口"""
    config = get_config()
    app = DarwinApp(config)
    app.run()


if __name__ == "__main__":
    main()