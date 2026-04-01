"""
Darwin 统一主程序
集成 main_root_realtime.py 和 main_openmmlab.py 的功能
通过配置文件控制显示哪些窗口
"""
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config, HTTPConfig
from core.camera_capture import CameraCapture
from core.video_decoder import VideoDecoder
from core.video_encoder import VideoEncoder  # 确保在 video_encoder.py 中声明了类
from core.action_http_client import ActionHTTPClient
from core.action_websocket_client import ActionWebSocketClient
from core.http_client import HTTPClient
from core.mujoco_simulator import MuJoCoRobotSimulator
from core.display import Display
from utils.logger import get_logger


# 直接导入 VideoEncoder，避免延迟加载问题
try:
    from core.video_encoder import VideoEncoder as _VideoEncoderCheck
    print(f"VideoEncoder 类导入成功: {_VideoEncoderCheck}")
except ImportError as e:
    print(f"VideoEncoder 类导入失败: {e}")
    raise


class DarwinIntegratedApp:
    """Darwin集成应用 - 支持可选的多窗口显示"""

    def __init__(self, config: Config, log_dimension: int = 29):
        self.config = config
        self.log_dimension = log_dimension
        self._logger = get_logger(__name__, config.log_level)

        # 初始化相机模块
        self.camera = CameraCapture(config.camera)

        # 初始化视频编码器（新增异步编码）
        self.encoder = VideoEncoder(config.video, quality=75)

        # 初始化视频解码器（用于解码处理后的图像）
        self.decoder = VideoDecoder(config.video)

        # 初始化显示模块
        self.display = Display(config.display)

        # 根据配置初始化HTTP客户端或WebSocket客户端
        # 可以同时使用两个客户端
        self.action_client = None
        self.openmm_client = None

        if config.mujoco.show_mujoco:
            # 根据配置选择 HTTP 或 WebSocket
            if config.mujoco.http.use_websocket:
                self._logger.info("使用WebSocket模式连接动作服务")
                self.action_client = ActionWebSocketClient(config.mujoco.http)
                # 等待WebSocket连接
                for _ in range(50):
                    time.sleep(0.1)
                    if self.action_client.is_connected():
                        self._logger.info("WebSocket客户端连接已建立")
                        break
                else:
                    self._logger.error("WebSocket连接超时，请检查服务端是否运行")
            else:
                self._logger.info("使用HTTP模式连接动作服务")
                self.action_client = ActionHTTPClient(config.mujoco.http)

        if config.display.show_processed:
            self.openmm_client = HTTPClient(config.display.http)
            self._logger.info(f"图像处理客户端已初始化: {config.display.http.base_url}{config.display.http.endpoint}")

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
        self._last_frame_update = 0  # 用于限流显示更新

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

                if self._frame_count <= 3:
                    self._logger.info(f"[{self._frame_count}] 获取到相机帧: {frame.shape}")

                # 2. 异步编码帧数据用于传输
                encoded_frame = self.encoder.encode_frame(frame)

                # 前几帧打印编码状态
                if encoded_frame is None and self._frame_count <= 10:
                    self._logger.info(f"[{self._frame_count}] 编码还未完成...")
                elif encoded_frame is not None and self._frame_count <= 3:  # 前3帧只打印
                    self._logger.info(f"[{self._frame_count}] 编码完成！数据大小: {len(encoded_frame)} 字节")

                # 3. 根据配置调用HTTP客户端
                processed_frame = None
                action_data = None

                # 机器人控制模式（优先处理）
                self._logger.debug(f"[{self._frame_count}] 检查条件: show_mujoco={self.config.mujoco.show_mujoco}, action_client={self.action_client is not None}, encoded_frame={encoded_frame is not None}")

                # 判断是否需要人体检测
                should_process_robot = True
                if self.config.mujoco.require_human_detection:
                    should_process_robot = self._check_human_detection(frame)

                if self.config.mujoco.show_mujoco and self.action_client and encoded_frame and should_process_robot:
                    # 获取动作数据
                    self._logger.info(f"[{self._frame_count}] 开始发送帧到WebSocket, 大小: {len(encoded_frame)} 字节")
                    action_data = self.action_client.send_frame(encoded_frame)
                    if action_data:
                        self._logger.info(f"[{self._frame_count}] 收到动作数据: {list(action_data.keys())}")
                        # 新格式：motions.dof_pos (29个关节)
                        if 'motions' in action_data and 'dof_pos' in action_data['motions']:
                            dof_pos = action_data['motions']['dof_pos']
                            self._logger.info(f"[{self._frame_count}] 应用 dof_pos: {len(dof_pos)} 个关节")
                            self._apply_robot_action(dof_pos)
                        # 兼容旧格式：q
                        elif 'q' in action_data:
                            self._logger.info(f"[{self._frame_count}] 应用 q: {len(action_data['q'])} 个关节")
                            self._apply_robot_action(action_data['q'])
                        else:
                            self._logger.warning(f"[{self._frame_count}] 动作数据格式未知，keys: {list(action_data.keys())}")
                    else:
                        self._logger.warning(f"[{self._frame_count}] 未收到动作数据，可能服务端未响应")
                else:
                    if self._frame_count <= 5:
                        if not self.config.mujoco.show_mujoco:
                            self._logger.info(f"[{self._frame_count}] 控制被跳过: MuJoCo窗口未启用")
                        elif not self.action_client:
                            self._logger.info(f"[{self._frame_count}] 控制被跳过: action_client 未初始化")
                        elif not encoded_frame:
                            self._logger.info(f"[{self._frame_count}] 控制被跳过: encoded_frame 为空 (编码未完成)")

                # 图像处理模式（可同时进行）
                if self.config.display.show_processed and self.openmm_client:
                    # /process/frame_yolo 接口需要原始 numpy 字节流，不是 JPEG 编码
                    self._logger.debug("发送图像处理请求...")
                    result = self.openmm_client.send_frame(
                        frame.tobytes(),
                        height=frame.shape[0],
                        width=frame.shape[1],
                        channels=frame.shape[2] if len(frame.shape) == 3 else 1
                    )
                    processed_frame = self.decoder.decode_result(result)
                    if processed_frame is not None:
                        self._logger.debug("收到处理结果")
                    else:
                        self._logger.debug("处理结果为空")

                # 4. 根据配置显示窗口（非阻塞，放入队列）
                if self.config.display.show_original:
                    self.display.show_original(frame)

                if self.config.display.show_processed:
                    if processed_frame is not None:
                        self.display.show_processed(processed_frame)
                    elif self.config.mujoco.show_mujoco:
                        # 如果机器人模式开启但没有processed frame，显示原始帧
                        self.display.show_processed(frame)

                # 5. 检查退出信号（由显示线程处理）
                if self.display.check_exit():
                    self._logger.info("用户退出程序")
                    break

                # 检查MuJoCo窗口
                if self.simulator and not self.simulator.is_running():
                    self._logger.info("MuJoCo窗口已关闭")
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

    def _check_human_detection(self, frame):
        """
        检查图像中是否有人体

        Args:
            frame: 图像帧 (numpy array)

        Returns:
            bool: True表示检测到人体， False表示未检测到人体
        """
        try:
            # 创建配置给HTTP客户端使用
            http_config = HTTPConfig(
                base_url=self.config.mujoco.human_detection_base_url,
                endpoint=self.config.mujoco.human_detection_endpoint,
                timeout=5.0,
                max_retries=1
            )

            detection_client = HTTPClient(http_config)

            # 构造数据：前12字节为尺寸信息，后面是图像数据
            height, width = frame.shape[0], frame.shape[1]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1

            payload = frame.tobytes()

            # 调用人体检测API
            result = detection_client.detect_person(payload, height, width, channels)

            if result:
                # 解析检测结果
                has_person = result.get("has_person", False)
                has_complete_person = result.get("has_complete_person", False)
                person_count = result.get("person_count", 0)

                if has_person and has_complete_person:
                    self._logger.info(f"[{self._frame_count}] 检测到完整人体 (共{person_count}人)")
                elif has_person:
                    self._logger.info(f"[{self._frame_count}] 检测到人体但不完整")
                else:
                    self._logger.debug(f"[{self._frame_count}] 未检测到人体")

                return has_person and has_complete_person
            else:
                self._logger.debug(f"[{self._frame_count}] 人体检测失败，认为无人")
                return False

        except Exception as e:
            self._logger.error(f"[{self._frame_count}] 人体检测异常: {e}")
            # 检测出错时，保守处理，认为不安全，跳过调用
            return False

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

        if self.encoder:
            self.encoder.close()

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