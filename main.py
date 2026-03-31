"""
Darwin 统一主程序
集成 main_root_realtime.py 和 main_openmmlab.py 的功能
通过配置文件控制显示哪些窗口
"""
import cv2
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from core.camera_capture import CameraCapture
from core.video_decoder import VideoDecoder
from core.action_http_client import ActionHTTPClient
from core.http_client import HTTPClient
from core.mujoco_simulator import MuJoCoRobotSimulator
from core.display import Display
from utils.logger import get_logger


class DarwinIntegratedApp:
    """Darwin集成应用 - 支持可选的多窗口显示"""

    def __init__(self, config: Config, log_dimension: int = 29):
        self.config = config
        self.log_dimension = log_dimension
        self._logger = get_logger(__name__, config.log_level)

        # 初始化相机模块
        self.camera = CameraCapture(config.camera)

        # 初始化视频编码器
        self.decoder = VideoDecoder(config.video)

        # 初始化显示模块
        self.display = Display(config.display)

        # 根据配置初始化HTTP客户端
        # 可以同时使用两个HTTP客户端
        self.action_client = None
        self.openmm_client = None

        if config.mujoco.show_mujoco:
            self.action_client = ActionHTTPClient(config.mujoco.http)

        if config.display.show_processed:
            self.openmm_client = HTTPClient(config.display.http)

        # 如果两个都没启用，则显示错误
        if not self.action_client and not self.openmm_client:
            self._logger.error(
                "没有启用任何HTTP客户端，请检查配置中的 show_mujoco 或 show_processed"
            )
            raise RuntimeError("没有可用的HTTP客户端")

        # MuJoCo仿真器（根据配置初始化）
        self.simulator: Optional[MuJoCoRobotSimulator] = None
        self._mujoco_initialized = False

        if config.mujoco.show_mujoco:
            self._mujoco_initialized = self._init_mujoco()

            # 如果MuJoCo初始化失败，则禁用相关功能
            if not self._mujoco_initialized:
                self._logger.warning("MuJoCo仿真器初始化失败，禁用机器人控制功能")
                self.config.mujoco.show_mujoco = False
                if self.action_client:
                    self.action_client = None

        # 动作状态
        self._current_positions = None
        self._target_positions = None
        self._last_action_time = 0
        self._action_count = 0
        self._frame_count = 0

    def _init_mujoco(self) -> bool:
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
        self._logger.info("Darwin集成应用启动中...")

        # 打印当前配置的显示模式
        self._print_display_config()

        # 打开相机
        if not self.camera.open():
            self._logger.error("无法打开相机，程序退出")
            return

        # MuJoCo仿真准备阶段
        if self.simulator and self._mujoco_initialized:
            self._current_positions = self.simulator.get_joint_positions()
            self._prepare_robot()

        self._logger.info("开始实时控制循环...")
        print("\n" + "="*50)
        print("实时控制已启动")
        print("="*50)

        try:
            while True:
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

                # 3. 根据配置调用HTTP客户端
                processed_frame = None
                action_data = None

                # 机器人控制模式（优先处理）
                if self.config.mujoco.show_mujoco and self.action_client:
                    # 使用新接口 /process/frame，直接发送帧数据无需图片转换
                    action_data = self.action_client.send_frame(encoded_frame)
                    if action_data and 'q' in action_data:
                        self._apply_robot_action(action_data['q'])

                # 图像处理模式（可同时进行）
                if self.config.display.show_processed and self.openmm_client:
                    # /process/frame_yolo 接口需要原始 numpy 字节流，不是 JPEG 编码
                    result = self.openmm_client.send_frame(
                        frame.tobytes(),
                        height=frame.shape[0],
                        width=frame.shape[1],
                        channels=frame.shape[2] if len(frame.shape) == 3 else 1
                    )
                    processed_frame = self.decoder.decode_result(result)

                # 4. 根据配置显示窗口
                if self.config.display.show_original:
                    self.display.show_original(frame)

                if self.config.display.show_processed:
                    if processed_frame is not None:
                        self.display.show_processed(processed_frame)
                    elif self.config.mujoco.show_mujoco:
                        # 如果机器人模式开启但没有processed frame，显示原始帧
                        self.display.show_processed(frame)

                # 5. 检查退出按键
                key = self.display.wait_key(1)

                # 根据配置的窗口检查退出条件
                should_exit = False
                if key == 27:  # ESC键
                    should_exit = True

                # 检查原始窗口
                if self.config.display.show_original:
                    window_closed = (
                        cv2.getWindowProperty(self.config.display.window_name_original, cv2.WND_PROP_VISIBLE) < 1
                    )
                    if window_closed:
                        should_exit = True

                # 检查处理窗口
                if self.config.display.show_processed:
                    window_closed = (
                        cv2.getWindowProperty(self.config.display.window_name_processed, cv2.WND_PROP_VISIBLE) < 1
                    )
                    if window_closed:
                        should_exit = True

                # 检查MuJoCo窗口
                if self.simulator and not self.simulator.is_running():
                    should_exit = True

                if should_exit:
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

    def _apply_robot_action(self, joint_positions):
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

    def _print_display_config(self):
        """打印当前显示配置"""
        print("\n当前显示配置:")
        print("-" * 50)
        print(f"  原始相机窗口: {'启用' if self.config.display.show_original else '禁用'}")
        print(f"  处理后窗口:   {'启用' if self.config.display.show_processed else '禁用'}")
        print(f"  MuJoCo窗口:   {'启用' if self.config.mujoco.show_mujoco else '禁用'}")
        print(f"  运行模式:     {'机器人控制' if self.config.mujoco.show_mujoco else '模型处理'}")
        print("-" * 50)

        if self.config.mujoco.show_mujoco:
            print(f"动作服务: {self.config.mujoco.http.base_url}{self.config.mujoco.http.endpoint}")
            print(f"仿真机器人: {self.config.mujoco.robot} ({self.log_dimension} DOF)")
        else:
            print(f"模型服务: {self.config.display.http.base_url}{self.config.display.http.endpoint}")

        print(f"相机设备: {self.config.camera.device_id}")
        print()

    def _cleanup(self):
        """清理资源"""
        self._logger.info("正在清理资源...")
        self.camera.release()
        self.display.destroy()

        if self.action_client:
            self.action_client.close()

        if self.openmm_client:
            self.openmm_client.close()

        if self.simulator:
            self.simulator.cleanup()

        if self.config.mujoco.show_mujoco:
            self._logger.info(
                f"程序已退出，共处理 {self._action_count} 个动作请求"
            )
        else:
            self._logger.info("程序已退出")


def main():
    """程序入口"""
    print("=" * 50)
    print("Darwin 集成控制程序")
    print("=" * 50)
    print()

    config = get_config()

    # 默认使用29个自由度的G1机器人
    log_dimension = 29

    app = DarwinIntegratedApp(config, log_dimension)
    app.run()


if __name__ == "__main__":
    main()