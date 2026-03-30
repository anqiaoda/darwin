"""
Darwin 实时机器人控制程序
实时采集相机视频流->通过HTTP获取动作数据->驱动MuJoCo仿真机器人
结合 DarwinApp 的相机采集功能和 main_root.py 的机器人控制功能
"""
import cv2
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from core.camera_capture import CameraCapture
from core.video_decoder import VideoDecoder
from core.action_http_client import ActionHTTPClient
from core.mujoco_simulator import MuJoCoRobotSimulator
from core.display import Display
from utils.logger import get_logger


class RobotDarwinApp:
    """Darwin机器人实时控制应用"""

    def __init__(self, config: Config, log_dimension: int = 29):
        self.config = config
        self.log_dimension = log_dimension
        self._logger = get_logger(__name__, config.log_level)

        # 初始化相机模块
        self.camera = CameraCapture(config.camera)

        # 初始化视频编码器（用于发送图片到HTTP服务）
        self.decoder = VideoDecoder(config.video)

        # 初始化动作HTTP客户端
        self.action_client = ActionHTTPClient(config.action_http)

        # 初始化显示模块
        self.display = Display(config.display)

        # MuJoCo仿真器（稍后初始化）
        self.simulator: Optional[MuJoCoRobotSimulator] = None

        # 动作状态
        self._current_positions = None
        self._target_positions = None
        self._last_action_time = 0
        self._action_count = 0
        self._frame_count = 0

    def init_simulator(self) -> bool:
        """初始化MuJoCo仿真器"""
        try:
            self._logger.info("正在初始化MuJoCo仿真器...")
            self.simulator = MuJoCoRobotSimulator(
                self.config.mujoco,
                self.log_dimension
            )
            self.simulator.start_background_threads()
            self._logger.info("MuJoCo仿真器初始化成功")
            return True
        except Exception as e:
            self._logger.error(f"初始化仿真器失败: {e}", exc_info=True)
            return False

    def run(self):
        """运行主循环"""
        self._logger.info("Darwin机器人实时控制应用启动中...")

        # 打开相机
        if not self.camera.open():
            self._logger.error("无法打开相机，程序退出")
            return

        # 初始化仿真器
        if not self.init_simulator():
            self._logger.error("初始化仿真器失败，程序退出")
            return

        # 获取初始关节位置
        self._current_positions = self.simulator.get_joint_positions()
        self._logger.info(f"初始关节位置 (前5个): {self._current_positions[:5]}")

        # 准备阶段：平滑过渡到待机姿势
        self._prepare_robot()

        self._logger.info("开始实时控制循环...")
        print("\n" + "="*50)
        print("实时控制已启动")
        print("="*50)
        print(f"动作服务: {self.config.action_http.base_url}{self.config.action_http.endpoint}")
        print("窗口1: 原始相机画面")
        print("窗口2: MuJoCo机器人仿真 (独立窗口)")
        print("按 ESC 或关闭窗口退出\n")

        try:
            while True:
                loop_start = time.perf_counter()
                self._frame_count += 1

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

                # 3. 发送到HTTP动作服务获取动作数据
                action_data = self.action_client.send_image(encoded_frame)

                # 4. 应用动作到仿真机器人
                if action_data and 'q' in action_data:
                    self._robot_action_control_thread(action_data['q'])

                # 5. 显示原始相机画面
                self.display.show_original(frame)

                # 6. 检查退出按键
                key = self.display.wait_key(1)
                window_closed = (
                    cv2.getWindowProperty(self.config.display.window_name_original, cv2.WND_PROP_VISIBLE) < 1
                )
                if key == 27 or window_closed or (self.simulator and not self.simulator.is_running()):
                    self._logger.info("用户退出程序")
                    break

        except KeyboardInterrupt:
            self._logger.info("收到中断信号，退出程序")

        except Exception as e:
            self._logger.error(f"程序异常: {e}", exc_info=True)

        finally:
            self._cleanup()

    def _prepare_robot(self):
        """准备阶段：平滑过渡到零位"""
        if not self.simulator:
            return

        self._logger.info("准备阶段：平滑过渡到待机姿势...")

        target_positions = np.zeros(self.log_dimension)
        current_positions = self.simulator.get_joint_positions()

        for step in range(1, self.config.mujoco.prepare_steps + 1):
            if not self.simulator.is_running():
                break

            ratio = np.clip(step / self.config.mujoco.prepare_steps, 0.0, 1.0)
            interpolated = ratio * target_positions + (1 - ratio) * current_positions

            self.simulator.set_joint_positions(interpolated)
            time.sleep(self.config.mujoco.control_dt)

        self._current_positions = target_positions
        self._logger.info("准备阶段完成")

    def _robot_action_control_thread(self, joint_positions):
        """应用关节动作到机器人

        Args:
            joint_positions: 关节位置数组
        """
        if not self.simulator:
            return

        # 截取前 log_dimension 个关节
        n = min(len(joint_positions), self.log_dimension)
        if n == 0:
            return

        frame_data = joint_positions[:n]

        # 直接设置关节位置（运动学模式）
        self.simulator.set_joint_positions(frame_data)
        self._current_positions = frame_data
        self._action_count += 1

        # 每秒打印一次统计信息
        current_time = time.time()
        if current_time - self._last_action_time >= 1.0:
            self._logger.info(
                f"已处理 {self._action_count} 个动作请求, "
                f"帧率: {self._frame_count / (current_time - self._last_action_time):.1f} FPS"
            )
            self._last_action_time = current_time
            self._frame_count = 0

    def _cleanup(self):
        """清理资源"""
        self._logger.info("正在清理资源...")
        self.camera.release()
        self.display.destroy()
        self.action_client.close()

        if self.simulator:
            self.simulator.cleanup()

        self._logger.info(
            f"程序已退出，共处理 {self._action_count} 个动作请求"
        )


def main():
    """程序入口"""
    print("=" * 50)
    print("Darwin 机器人实时控制程序")
    print("=" * 50)
    print()

    config = get_config()

    # 默认使用29个自由度的G1机器人
    log_dimension = 29

    print(f"动作服务: {config.action_http.base_url}{config.action_http.endpoint}")
    print(f"仿真机器人: {config.mujoco.robot} ({log_dimension} DOF)")
    print(f"相机设备: {config.camera.device_id}")
    print()

    app = RobotDarwinApp(config, log_dimension)
    app.run()


if __name__ == "__main__":
    main()