"""
配置管理模块
集中管理系统配置参数，支持从config.json读取
"""
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class CameraConfig:
    """相机配置"""
    device_id: int = 0        # 相机设备ID
    width: int = 640          # 视频宽度
    height: int = 480         # 视频高度
    fps: int = 30             # 帧率
    buffer_size: int = 1      # 缓冲区大小
    backend: str = "dshow"    # 后端类型(dshow/autodetect)


@dataclass
class HTTPConfig:
    """HTTP服务配置"""
    base_url: str = "http://172.18.20.118:9000"  # 模型服务基础URL
    endpoint: str = "/process/image"              # API端点
    timeout: int = 10                             # 请求超时时间(秒)
    max_retries: int = 3                          # 最大重试次数
    use_websocket: bool = False                   # 是否使用WebSocket
    websocket_endpoint: str = "/ws/infer"         # WebSocket端点（可选）


@dataclass
class VideoConfig:
    """视频配置"""
    codec: str = "mp4v"                # 编解码器
    output_width: Optional[int] = None  # 输出宽度(None=保持原尺寸)
    output_height: Optional[int] = None # 输出高度


@dataclass
class RuntimeConfig:
    """运行时显示配置（已废弃，配置已移至各自模块）"""
    pass


@dataclass
class DisplayConfig:
    """显示配置"""
    window_name_original: str = "Darwin Original"    # 原始视频窗口名称
    window_name_processed: str = "Darwin Processed"  # 处理后视频窗口名称
    show_fps: bool = True                           # 显示帧率
    scale_factor: float = 1.0                       # 显示缩放因子
    show_original: bool = True                      # 显示原始相机窗口
    show_processed: bool = True                     # 显示处理后的窗口（骨骼点渲染等）
    http: HTTPConfig = field(default_factory=HTTPConfig)  # HTTP服务配置


@dataclass
class MuJoCoConfig:
    """MuJoCo 仿真配置"""
    robot: str = "g1"                              # 机器人型号
    scene_file: Optional[str] = None               # 场景文件路径 (None=自动推断)
    simulate_dt: float = 0.005                     # 仿真时间步长
    viewer_dt: float = 0.02                        # 渲染器时间步长
    control_dt: float = 0.02                       # 控制时间步长
    prepare_steps: int = 60                        # 准备阶段帧数
    gravity: list = field(default_factory=lambda: [0, 0, 0])  # 重力设置
    show_mujoco: bool = False                      # 显示MuJoCo机器人仿真窗口
    http: HTTPConfig = field(default_factory=HTTPConfig)  # HTTP服务配置


@dataclass
class Config:
    """系统总配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    mujoco: MuJoCoConfig = field(default_factory=MuJoCoConfig)
    log_level: str = "INFO"              # 日志级别
    temp_dir: str = "./data/temp"        # 临时目录


def get_config_path() -> Path:
    """获取配置文件路径"""
    # 优先使用exe所在目录的config.json
    if getattr(sys, 'frozen', False):
        # 打包后的exe环境
        exe_dir = Path(sys.executable).parent
        config_path = exe_dir / "config.json"
        if config_path.exists():
            return config_path

    # 开发环境，使用config.py所在目录
    return Path(__file__).parent / "config.json"


def load_config_from_file(config_path: Optional[Path] = None) -> dict:
    """
    从JSON文件加载配置

    Args:
        config_path: 配置文件路径，默认为config.json

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def get_http_config(http_data: dict) -> HTTPConfig:
    """从JSON数据创建HTTP配置对象

    Args:
        http_data: HTTP配置字典

    Returns:
        HTTPConfig实例
    """
    return HTTPConfig(
        base_url=http_data.get("base_url", "http://172.18.20.118:9000"),
        endpoint=http_data.get("endpoint", "/process/image"),
        timeout=http_data.get("timeout", 10),
        max_retries=http_data.get("max_retries", 3),
        use_websocket=http_data.get("use_websocket", False),
        websocket_endpoint=http_data.get("websocket_endpoint", "/ws/infer")
    )


def get_config(config_file: Optional[str] = None) -> Config:
    """
    获取配置实例

    Args:
        config_file: 配置文件路径

    Returns:
        配置实例
    """
    config_path = Path(config_file) if config_file else get_config_path()
    config_data = load_config_from_file(config_path)

    # 从JSON数据创建配置对象
    camera = CameraConfig(**config_data.get("camera", {}))
    video = VideoConfig(**config_data.get("video", {}))

    display_data = config_data.get("display", {})
    display_http = get_http_config(display_data.get("http", {}))
    display = DisplayConfig(
        window_name_original=display_data.get("window_name_original", "Darwin Original"),
        window_name_processed=display_data.get("window_name_processed", "Darwin Processed"),
        show_fps=display_data.get("show_fps", True),
        scale_factor=display_data.get("scale_factor", 1.0),
        show_original=display_data.get("show_original", True),
        show_processed=display_data.get("show_processed", True),
        http=display_http
    )

    mujoco_data = config_data.get("mujoco", {})
    mujoco_http = get_http_config(mujoco_data.get("http", {}))
    gravity = mujoco_data.get("gravity", [0, 0, 0])

    mujoco = MuJoCoConfig(
        robot=mujoco_data.get("robot", "g1"),
        scene_file=mujoco_data.get("scene_file"),
        simulate_dt=mujoco_data.get("simulate_dt", 0.005),
        viewer_dt=mujoco_data.get("viewer_dt", 0.02),
        control_dt=mujoco_data.get("control_dt", 0.02),
        prepare_steps=mujoco_data.get("prepare_steps", 60),
        gravity=gravity,
        show_mujoco=mujoco_data.get("show_mujoco", False),
        http=mujoco_http
    )

    log_level = config_data.get("log_level", "INFO")
    temp_dir = config_data.get("temp_dir", "./data/temp")

    return Config(
        camera=camera,
        video=video,
        display=display,
        mujoco=mujoco,
        log_level=log_level,
        temp_dir=temp_dir
    )