"""
MuJoCo 机器人仿真模拟器模块
封装 MuJoCo 仿真环境，提供实时机器人控制接口
纯运动学模式：直接写 qpos + mj_forward，不使用物理仿真
支持关节角度插值（LERP）和根节点位置/旋转插值（LERP + SLERP）
"""
import time
import numpy as np
from threading import Thread, Lock, Event
from pathlib import Path
from typing import Optional

from config import MuJoCoConfig
from utils.logger import get_logger


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值（四元数 scalar_first 格式 [w, x, y, z]）

    Args:
        q1: 起始四元数 [w, x, y, z]
        q2: 目标四元数 [w, x, y, z]
        t: 插值因子 [0, 1]

    Returns:
        插值后的四元数
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2

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
    """MuJoCo 机器人模拟器（纯运动学模式，含插值）"""

    def __init__(self, config: MuJoCoConfig, log_dimension=29):
        """初始化仿真环境

        Args:
            config: MuJoCo 配置
            log_dimension: 日志数据中的关节数（29 或 23）
        """
        self.config = config
        self.log_dimension = log_dimension
        self._lock = Lock()
        self._interp_lock = Lock()
        self._logger = get_logger(__name__)

        # 延迟导入 MuJoCo
        _import_mujoco()

        # 确定场景文件路径
        if config.scene_file:
            scene_path = Path(config.scene_file)
        else:
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

        # 获取关节数量信息（qpos 布局：[pos(3), rot(4), joints(N)]）
        self.num_joints = self.mj_model.nq - 7  # 减去根节点（pos 3 + rot 4）
        self._logger.info(f"机器人有 {self.num_joints} 个关节")
        self._logger.info(f"日志数据维度: {log_dimension}")

        # 启动查看器（隐藏 UI 面板，只显示 3D 场景）
        self.viewer = _mujoco_viewer.launch_passive(
            self.mj_model, self.mj_data,
            show_left_ui=False,
            show_right_ui=False
        )

        # 隐藏碰撞几何体（去掉地板等 geomgroup 0）
        self.viewer.opt.geomgroup[0] = False

        # 线程控制
        self._sim_thread: Optional[Thread] = None
        self._stop_event = Event()

        # 插值状态（由 _interp_lock 保护）
        self._former_positions: Optional[np.ndarray] = None
        self._target_positions: Optional[np.ndarray] = None
        self._interp_duration = 0.0
        self._current_interp_time = 0.0
        self._is_interpolating = False
        self._max_interp_time = config.max_interp_time
        self._delta_time = 0.001  # 插值推进步长（秒）
        self._former_time = None  # 上次收到目标数据的时间

        # 预分配插值用缓冲区（避免每次调用创建临时数组）
        self._padded = np.zeros(self.num_joints, dtype=np.float64)

        # 根节点插值状态（由 _interp_lock 保护）
        self._former_root_pos: Optional[np.ndarray] = None
        self._target_root_pos: Optional[np.ndarray] = None
        self._former_root_rot: Optional[np.ndarray] = None
        self._target_root_rot: Optional[np.ndarray] = None
        self._has_root_target = False  # 是否有根节点目标数据

        time.sleep(0.2)

    def set_joint_positions(self, target_positions):
        """直接设置关节位置（运动学模式，无插值）

        用于准备阶段等需要直接设置的场景。

        Args:
            target_positions: 目标关节位置数组
        """
        with self._lock:
            self._apply_positions(target_positions)

    def set_target_positions(self, target_positions, root_pos=None, root_rot=None):
        """设置目标关节位置和根节点状态

        如果启用插帧（enable_interpolation=True），触发插值过渡。
        如果禁用插帧，直接应用数据到机器人。

        Args:
            target_positions: 目标关节位置数组
            root_pos: 根节点目标位置 [x, y, z]（可选）
            root_rot: 根节点目标旋转 [w, x, y, z] scalar_first 格式（可选）
        """
        if not self.config.enable_interpolation:
            # 无插帧模式：直接应用
            with self._lock:
                n = min(len(target_positions), self.log_dimension, self.num_joints)
                positions = np.array(target_positions[:n], dtype=np.float64)
                if root_pos is not None and root_rot is not None:
                    self._apply_full_state(
                        np.array(root_pos, dtype=np.float64),
                        np.array(root_rot, dtype=np.float64),
                        positions
                    )
                else:
                    self._apply_positions(positions)
            return

        # 插帧模式
        with self._interp_lock:
            # 如果正在插值，将 former 更新为当前插值位置（平滑过渡）
            if self._is_interpolating and self._former_positions is not None and self._target_positions is not None:
                t = self._current_interp_time / self._interp_duration if self._interp_duration > 0 else 1.0
                t = np.clip(t, 0.0, 1.0)
                self._former_positions = (1 - t) * self._former_positions + t * self._target_positions
                # 根节点插值中点
                if self._has_root_target and self._former_root_pos is not None:
                    self._former_root_pos = (1 - t) * self._former_root_pos + t * self._target_root_pos
                    self._former_root_rot = slerp(self._former_root_rot, self._target_root_rot, t)
            else:
                # 首次设置或插值已完成，从当前实际位置开始
                self._former_positions = self._read_current_positions_unsafe()

            # 设置新目标
            n = min(len(target_positions), self.log_dimension, self.num_joints)
            self._target_positions = np.array(target_positions[:n], dtype=np.float64)

            # 补齐到 num_joints（用 former 对应位置的值，复用预分配缓冲区）
            if len(self._target_positions) < self.num_joints and self._former_positions is not None:
                self._padded[:len(self._target_positions)] = self._target_positions
                self._padded[len(self._target_positions):] = self._former_positions[len(self._target_positions):]
                self._target_positions = self._padded.copy()

            # 同样补齐 former
            if self._former_positions is not None and len(self._former_positions) < self.num_joints:
                self._padded[:len(self._former_positions)] = self._former_positions
                self._padded[len(self._former_positions):] = 0
                self._former_positions = self._padded.copy()

            # 根节点目标
            if root_pos is not None and root_rot is not None:
                root_pos = np.array(root_pos, dtype=np.float64)
                root_rot = np.array(root_rot, dtype=np.float64)

                if not self._has_root_target or self._former_root_pos is None:
                    # 首次设置根节点，从当前 qpos 读取
                    self._former_root_pos = self.mj_data.qpos[:3].copy()
                    self._former_root_rot = self.mj_data.qpos[3:7].copy()

                self._target_root_pos = root_pos
                self._target_root_rot = root_rot
                self._has_root_target = True
            else:
                # 没有根节点数据时，不更新根节点目标
                self._has_root_target = False

            # 插值时长：根据实际帧间隔动态计算
            current_time = time.time()
            if self._former_time is not None:
                time_diff = current_time - self._former_time
                self._interp_duration = min(time_diff, self._max_interp_time)
            else:
                self._interp_duration = self._max_interp_time
            self._former_time = current_time

            self._current_interp_time = 0.0
            self._is_interpolating = True

    def _read_current_positions_unsafe(self):
        """读取当前关节位置（不加锁，仅用于 _interp_lock 内部调用）

        Returns:
            关节位置数组 (长度为 num_joints)
        """
        n = min(self.log_dimension, self.num_joints)
        return self.mj_data.qpos[7:7+n].copy()

    def _step_interpolation(self, need_render=False):
        """在仿真步中推进插值（在仿真线程中调用，已持有 _lock）

        每次仿真步调用，推进插值进度并更新 qpos。
        只在 need_render=True 时调用 mj_forward，减少不必要的计算。

        Args:
            need_render: 是否即将渲染，为 True 时才调用 mj_forward
        """
        with self._interp_lock:
            if not self._is_interpolating or self._former_positions is None or self._target_positions is None:
                return

            # 计算插值因子
            t = self._current_interp_time / self._interp_duration if self._interp_duration > 0 else 1.0
            t = np.clip(t, 0.0, 1.0)

            # 线性插值关节
            interp_positions = (1 - t) * self._former_positions + t * self._target_positions

            if self._has_root_target and self._former_root_pos is not None:
                # 插值根节点位置（LERP）
                interp_root_pos = (1 - t) * self._former_root_pos + t * self._target_root_pos
                # 插值根节点旋转（SLERP）
                interp_root_rot = slerp(self._former_root_rot, self._target_root_rot, t)

                # 写入 qpos（不调用 mj_forward）
                self.mj_data.qpos[:3] = interp_root_pos
                self.mj_data.qpos[3:7] = interp_root_rot
                n = min(len(interp_positions), self.num_joints)
                self.mj_data.qpos[7:7+n] = interp_positions[:n]
                self.mj_data.qvel[:] = 0

                if need_render:
                    _mujoco.mj_forward(self.mj_model, self.mj_data)
            else:
                # 写入 qpos（不调用 mj_forward）
                n = min(len(interp_positions), self.num_joints)
                self.mj_data.qpos[7:7+n] = interp_positions[:n]
                self.mj_data.qvel[:] = 0

                if need_render:
                    _mujoco.mj_forward(self.mj_model, self.mj_data)

            # 推进插值时间
            self._current_interp_time += self._delta_time

            # 插值完成
            if self._current_interp_time >= self._interp_duration:
                self._former_positions = self._target_positions.copy()
                if self._has_root_target:
                    self._former_root_pos = self._target_root_pos.copy()
                    self._former_root_rot = self._target_root_rot.copy()
                self._is_interpolating = False

    def _apply_positions(self, positions):
        """将关节位置应用到 MuJoCo 模型（需在 _lock 内调用）

        直接写 qpos[7:]，不通过 actuator 映射。

        Args:
            positions: 关节位置数组
        """
        n = min(len(positions), self.num_joints)
        self.mj_data.qpos[7:7+n] = positions[:n]

        # 清零关节速度
        self.mj_data.qvel[:] = 0

        # 前向运动学计算
        _mujoco.mj_forward(self.mj_model, self.mj_data)

    def _apply_full_state(self, root_pos, root_rot, dof_positions):
        """将完整状态（根节点 + 关节）应用到 MuJoCo 模型（需在 _lock 内调用）

        参考 robot_motion_viewer.py 的 step() 方法：
        - qpos[:3] = root_pos
        - qpos[3:7] = root_rot (scalar_first)
        - qpos[7:] = dof_pos

        Args:
            root_pos: 根节点位置 [x, y, z]
            root_rot: 根节点旋转 [w, x, y, z] scalar_first 格式
            dof_positions: 关节位置数组
        """
        # 写入根节点位置（freejoint 的前 3 个元素）
        self.mj_data.qpos[:3] = root_pos

        # 写入根节点旋转（freejoint 的 4-7 元素，scalar_first [w,x,y,z]）
        self.mj_data.qpos[3:7] = root_rot

        # 写入关节角度
        n = min(len(dof_positions), self.num_joints)
        self.mj_data.qpos[7:7+n] = dof_positions[:n]

        # 清零关节速度
        self.mj_data.qvel[:] = 0

        # 前向运动学计算
        _mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_joint_positions(self):
        """获取当前关节位置

        Returns:
            关节位置数组 (长度为 log_dimension)
        """
        n = min(self.log_dimension, self.num_joints)
        positions = np.zeros(n)
        with self._lock:
            positions[:] = self.mj_data.qpos[7:7+n]
        return positions

    def step_simulation(self, need_render=False):
        """执行一次仿真步（纯运动学模式）

        只在插值进行中时推进插值并更新 qpos，
        need_render=True 时才调用 mj_forward 以减少计算开销。
        不调用 mj_step，避免物理仿真干扰运动学控制。
        """
        with self._lock:
            if self._is_interpolating:
                self._step_interpolation(need_render)

    def is_running(self):
        """检查查看器是否仍在运行"""
        return self.viewer.is_running()

    def start_background_threads(self):
        """启动后台仿真线程（含渲染）"""
        self._logger.info("启动后台仿真+渲染线程...")
        self._sim_thread = Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def stop_background_threads(self):
        """停止后台线程"""
        self._stop_event.set()
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)

    def _simulation_loop(self):
        """仿真+渲染循环（在后台线程中运行）

        插值以 delta_time 步长推进，保证动作流畅。
        渲染按 60 FPS 上限刷新，降低 GPU 负载。
        """
        self._logger.info("仿真+渲染线程启动")
        last_render_time = 0.0
        render_interval = 1.0 / 60.0  # 60 FPS 渲染上限

        # FPS 统计
        fps_frame_count = 0
        fps_last_time = time.time()

        while not self._stop_event.is_set() and self.is_running():
            # 判断是否即将渲染
            now = time.time()
            should_render = now - last_render_time >= render_interval

            # 插值推进（只在即将渲染时调用 mj_forward）
            self.step_simulation(need_render=should_render)

            # 渲染按固定频率刷新
            if should_render:
                with self._lock:
                    self.viewer.sync()
                last_render_time = now

                # FPS 统计（每秒打印一次）
                fps_frame_count += 1
                if now - fps_last_time >= 1.0:
                    fps = fps_frame_count / (now - fps_last_time)
                    self._logger.info(f"[MuJoCo] 渲染 FPS: {fps:.1f}")
                    fps_frame_count = 0
                    fps_last_time = now

            time.sleep(self._delta_time)

        self._logger.info("仿真+渲染线程结束")

    def cleanup(self):
        """清理资源"""
        self._logger.info("正在清理仿真器...")
        self.stop_background_threads()
