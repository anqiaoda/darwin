"""
MuJoCo 机器人仿真模拟器模块
封装 MuJoCo 仿真环境，提供实时机器人控制接口
"""
import time
import numpy as np
from threading import Thread, Lock, Event
from pathlib import Path

from config import MuJoCoConfig
from utils.logger import get_logger

# 延迟导入 MuJoCo，避免未启用时报错
_mujoco = None
_mujoco_viewer = None


def _import_mujoco():
    """延迟导入 MuJoCo 模块"""
    global _mujoco, _mujoco_viewer
    if _mujoco is None:
        try:
            import mujoco
            import mujoco.viewer
            _mujoco = mujoco
            _mujoco_viewer = mujoco.viewer
        except ImportError as e:
            raise ImportError(
                "MuJoCo 未正确安装或配置。请确保已安装 MuJoCo 及其依赖。\n"
                "安装方法: pip install mujoco\n"
                f"错误详情: {e}"
            )


class MuJoCoRobotSimulator:
    """MuJoCo 机器人模拟器"""

    def __init__(self, config: MuJoCoConfig, log_dimension=29):
        """初始化仿真环境

        Args:
            config: MuJoCo 配置
            log_dimension: 日志数据中的关节数（29 或 23）
        """
        self.config = config
        self.log_dimension = log_dimension
        self._lock = Lock()
        self._logger = get_logger(__name__)

        # 延迟导入 MuJoCo
        _import_mujoco()

        # 确定场景文件路径
        if config.scene_file:
            scene_path = Path(config.scene_file)
        else:
            # 使用darwin目录的父目录作为基准
            darwin_dir = Path(__file__).parent.parent
            scene_path = darwin_dir / "unitree_robots" / config.robot / "scene2.xml"

        if not scene_path.exists():
            raise FileNotFoundError(f"找不到场景文件: {scene_path}")

        self.scene_path = str(scene_path)
        self._logger.info(f"加载场景文件: {self.scene_path}")

        # 加载 MuJoCo 模型
        self.mj_model = _mujoco.MjModel.from_xml_path(self.scene_path)
        self.mj_data = _mujoco.MjData(self.mj_model)

        # 设置仿真参数
        self.mj_model.opt.timestep = config.simulate_dt

        # 设置重力
        self.mj_model.opt.gravity[:] = config.gravity

        # 获取关节和执行器信息
        self.num_actuators = self.mj_model.nu
        self._logger.info(f"机器人有 {self.num_actuators} 个执行器")
        self._logger.info(f"日志数据维度: {log_dimension}")

        # 构建执行器到关节的映射
        self.actuator_to_joint = {}
        for i in range(self.mj_model.nu):
            jnt_id = self.mj_model.actuator_trnid[i, 0]
            joint_name = self.mj_model.joint(jnt_id).name
            self.actuator_to_joint[i] = jnt_id
            if i < 35:
                self._logger.debug(f"  执行器 {i:2d} -> 关节 {jnt_id:2d}: {joint_name}")

        # 启动查看器
        self.viewer = _mujoco_viewer.launch_passive(self.mj_model, self.mj_data)

        # 隐藏 UI 面板，只显示 3D 场景
        self.viewer._sim().ui0_enable = False  # 左侧面板
        self.viewer._sim().ui1_enable = False  # 右侧面板

        # 线程控制
        self._sim_thread: Optional[Thread] = None
        self._viewer_thread: Optional[Thread] = None
        self._stop_event = Event()

        time.sleep(0.2)

    def set_joint_positions(self, target_positions):
        """直接设置关节位置（运动学模式）

        Args:
            target_positions: 目标关节位置数组 (长度为 log_dimension 或 num_actuators)
        """
        with self._lock:
            # 如果目标位置维度与执行器数量匹配，直接应用
            if len(target_positions) == self.num_actuators:
                for i in range(self.mj_model.nu):
                    jnt_id = self.actuator_to_joint[i]
                    qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
                    self.mj_data.qpos[qpos_adr] = target_positions[i]
            else:
                # 否则，只应用前 log_dimension 个关节
                n = min(len(target_positions), self.num_actuators)
                for i in range(n):
                    jnt_id = self.actuator_to_joint[i]
                    qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
                    self.mj_data.qpos[qpos_adr] = target_positions[i]

                # 其余关节保持当前位置
                for i in range(n, self.num_actuators):
                    jnt_id = self.actuator_to_joint[i]
                    qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
                    # 保持当前位置不变

            # 清零关节速度
            self.mj_data.qvel[:] = 0

            # 前向运动学计算
            _mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_joint_positions(self):
        """获取当前关节位置

        Returns:
            关节位置数组 (长度为 log_dimension)
        """
        positions = np.zeros(min(self.log_dimension, self.num_actuators))
        with self._lock:
            n = min(self.log_dimension, self.num_actuators)
            for i in range(n):
                jnt_id = self.actuator_to_joint[i]
                qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
                positions[i] = self.mj_data.qpos[qpos_adr]
        return positions

    def step_simulation(self):
        """执行一次仿真步"""
        with self._lock:
            _mujoco.mj_step(self.mj_model, self.mj_data)

    def sync_viewer(self):
        """同步查看器显示"""
        with self._lock:
            self.viewer.sync()

    def is_running(self):
        """检查查看器是否仍在运行"""
        return self.viewer.is_running()

    def start_background_threads(self):
        """启动后台仿真和渲染线程"""
        self._logger.info("启动后台仿真线程...")
        self._sim_thread = Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

        self._logger.info("启动后台渲染线程...")
        self._viewer_thread = Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    def stop_background_threads(self):
        """停止后台线程"""
        self._stop_event.set()
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)
        if self._viewer_thread:
            self._viewer_thread.join(timeout=2.0)

    def _simulation_loop(self):
        """仿真循环（在后台线程中运行）"""
        self._logger.info("仿真线程启动")
        while not self._stop_event.is_set() and self.is_running():
            step_start = time.perf_counter()

            self.step_simulation()

            # 控制仿真时间步长
            elapsed = time.perf_counter() - step_start
            time_until_next = self.config.simulate_dt - elapsed
            if time_until_next > 0:
                time.sleep(time_until_next)

        self._logger.info("仿真线程结束")

    def _viewer_loop(self):
        """渲染循环（在后台线程中运行）"""
        self._logger.info("渲染器线程启动")
        while not self._stop_event.is_set() and self.is_running():
            self.sync_viewer()
            time.sleep(self.config.viewer_dt)

        self._logger.info("渲染器线程结束")

    def cleanup(self):
        """清理资源"""
        self._logger.info("正在清理仿真器...")
        self.stop_background_threads()