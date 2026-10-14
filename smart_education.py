
"""
智能多学生分析 - 边缘卸载终极版
功能：
- 智能边缘卸载（DAO策略）
- 自动检测性能，动态选择策略
- 支持举手、玩手机等姿态分析
- 按 Q 或 ESC 退出
"""
import sys
import os
import time
import json
import cv2
import numpy as np
import grpc
import psutil
import platform
import tempfile
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_MODE_CONFIG = {
    "smooth": {"label": "Fluent", "pose_interval": 5},
    "standard": {"label": "Standard", "pose_interval": 3},
    "accurate": {"label": "High accuracy", "pose_interval": 1},
}
YOLO_CONFIG_DIR = Path(tempfile.gettempdir()) / "edge-video-ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))
import grpc_gen.inference_service_pb2 as pb2
import grpc_gen.inference_service_pb2_grpc as pb2_grpc

try:
    from src.dag.builder import build_education_pipeline
    from src.offload.base import NodeLoad
    from src.offload.dao import DAGAwareOffloadPolicy
except Exception as exc:
    build_education_pipeline = None
    NodeLoad = None
    DAGAwareOffloadPolicy = None
    logger.warning(f"DAO实时策略模块加载失败，将使用轻量规则策略: {exc}")

# 论文 DAO Algorithm 4.1 — 优先引用 raspi_client 中与论文一致的实现，
# 若外部模块不可用则自动回退到下方内置实现，确保 dao_policy 永不为 None。
_raspi_path = str(Path(__file__).resolve().parent.parent / "raspi_client")
ThesisDaoPolicy = None
if Path(_raspi_path).is_dir():
    if _raspi_path not in sys.path:
        sys.path.insert(0, _raspi_path)
    try:
        from policies.dao import DaoPolicy as ThesisDaoPolicy
    except Exception as _exc:
        logger.warning(f"外部 DaoPolicy 加载失败，将使用内置实现: {_exc}")

if ThesisDaoPolicy is None:
    import threading as _threading

    class ThesisDaoPolicy:  # noqa: F811 — 内置 fallback，与 raspi_client/policies/dao.py 算法一致
        """论文 Algorithm 4.1 内置实现（AIMD 拥塞控制 + 三层级联决策）。"""

        def __init__(
            self,
            deadline_ms: float = 200.0,
            max_server_workers: int = 2,
            local_pool_size: int = 2,
            initial_max_pending: int = 2,
            ema_alpha: float = 0.3,
        ):
            self._lock = _threading.Lock()
            self._deadline_ms = float(deadline_ms)
            self._max_server_workers = max(1, int(max_server_workers))
            self._local_pool_size = max(1, int(local_pool_size))
            self._ema_alpha = float(ema_alpha)
            self._pending_offload = 0
            self._pending_local = 0
            self._offload_ema = 100.0
            self._local_ema = 500.0
            self._max_pending_offload = max(1, min(int(initial_max_pending), 4))
            self._aimd_accum = 0.0

        def decide(self) -> bool:
            with self._lock:
                if self._pending_offload >= self._max_pending_offload:
                    return False
                if self._pending_local >= self._local_pool_size:
                    return True
                offload_cost = self._offload_ema * (
                    1.0 + self._pending_offload / self._max_server_workers
                )
                local_cost = self._local_ema * (
                    1.0 + self._pending_local / self._local_pool_size
                )
                return offload_cost < local_cost

        def enter_offload_queue(self) -> None:
            with self._lock:
                self._pending_offload += 1

        def enter_local_queue(self) -> None:
            with self._lock:
                self._pending_local += 1

        def report_offload_result(self, round_trip_ms: float) -> None:
            with self._lock:
                self._pending_offload = max(0, self._pending_offload - 1)
                self._offload_ema = (
                    self._ema_alpha * round_trip_ms
                    + (1.0 - self._ema_alpha) * self._offload_ema
                )
                if round_trip_ms <= self._deadline_ms:
                    self._aimd_accum += 0.1
                    if self._aimd_accum >= 1.0:
                        self._aimd_accum = 0.0
                        self._max_pending_offload = min(self._max_pending_offload + 1, 4)
                else:
                    self._aimd_accum = 0.0
                    self._max_pending_offload = max((self._max_pending_offload + 1) // 2, 1)

        def report_local_result(self, latency_ms: float) -> None:
            with self._lock:
                self._pending_local = max(0, self._pending_local - 1)
                self._local_ema = (
                    self._ema_alpha * latency_ms
                    + (1.0 - self._ema_alpha) * self._local_ema
                )

        def cancel_offload(self) -> None:
            with self._lock:
                self._pending_offload = max(0, self._pending_offload - 1)

    logger.info("使用内置 DaoPolicy 实现（AIMD 三层级联决策）")


class DeviceProfiler:
    """设备性能分析器"""
    
    @staticmethod
    def get_cpu_percent(interval=0.5):
        return psutil.cpu_percent(interval=interval)
    
    @staticmethod
    def get_memory_percent():
        return psutil.virtual_memory().percent
    
    @staticmethod
    def estimate_performance_score():
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_percent = psutil.virtual_memory().percent
        
        score = (cpu_count * 10) * (1 - cpu_percent/100) * (1 - mem_percent/100)
        return min(score, 100)
    
    @staticmethod
    def print_profile():
        logger.info(f"设备: {platform.node()}")
        logger.info(f"CPU核心: {psutil.cpu_count(logical=True)}, 使用率: {psutil.cpu_percent(interval=1):.1f}%")
        logger.info(f"内存: {psutil.virtual_memory().percent:.1f}%")
        logger.info(f"性能分数: {DeviceProfiler.estimate_performance_score():.1f}/100")


class SmartEducationOffload:
    def __init__(self, edge_host='192.168.10.22', edge_port=50051):
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.channel = None
        self.stub = None
        self.strategy = 'local'
        self.edge_connected = False  # 新增：真正连接状态
        self.analysis_mode = "standard"
        self.pose_interval = ANALYSIS_MODE_CONFIG[self.analysis_mode]["pose_interval"]
        self._cached_pose_keypoints = []
        self._cached_pose_count = None
        self._last_object_detections = []
        self._force_offload_frames = int(os.getenv("DAO_FORCE_OFFLOAD_FRAMES", "8"))
        # gRPC 容错：单次超时不彻底断连，连续失败达阈值才断开，避免需手动 curl 恢复
        self._rpc_timeout = float(os.getenv("EDGE_RPC_TIMEOUT_SECONDS", "30"))
        self._consecutive_offload_failures = 0
        self._offload_fail_threshold = int(os.getenv("EDGE_OFFLOAD_FAIL_THRESHOLD", "3"))
        self._last_reconnect_ts = 0.0
        self._reconnect_interval = float(os.getenv("EDGE_RECONNECT_INTERVAL_SEC", "3"))
        # gRPC 稳定性增强：keepalive 保活 + 探测超时放宽 + 通道复用（WiFi 抖动不再频繁断连）
        self._connect_probe_timeout = float(os.getenv("EDGE_CONNECT_PROBE_TIMEOUT", "10"))
        self._keepalive_interval = float(os.getenv("EDGE_KEEPALIVE_SEC", "30"))
        self._keepalive_timeout = float(os.getenv("EDGE_KEEPALIVE_TIMEOUT_SEC", "10"))
        self._channel_target = None  # 当前 channel 绑定的目标地址，地址变化时才重建
        self.dao_policy = self._create_dao_policy()
        self.dao_decision_info = {
            "enabled": self.dao_policy is not None,
            "policy": "DaoPolicy" if self.dao_policy else "fallback-rule",
            "strategy": "local",
            "reason": "尚未执行 DAO 决策",
            "admission": "pending",
            "priority": 0.0,
            "stage_count": 5,
            "target_node_id": None,
            "stages_to_offload": [],
            "recent_offloads": 0,
            "congestion_window_ms": None,
            "max_offloads_per_window": None,
            "estimated_transfer_ms": None,
            "is_gpu_heavy": None,
            "neighbors": [],
            "edge_queue": None,
            "edge_latency_ms": None,
            "local_cpu": None,
            "local_score": None,
            "pending_offload": 0,
            "pending_local": 0,
            "max_pending_offload": 2,
            "aimd_accum": 0.0,
            "offload_ema": 100.0,
            "local_ema": 500.0,
            "execution_note": "论文 DAO Algorithm 4.1: 三层级联决策 + AIMD 管道宽度控制。",
        }
        
        # 只有提供了有效的服务器地址才连接
        if edge_host and edge_port:
            self._connect()
        else:
            logger.warning("未配置边缘服务器，使用本地模式")
        
        # 本地模型
        self.yolo_model = None
        self.pose_model = None
        self.pipeline = None
        self._init_local_models()

    def set_analysis_mode(self, mode: str | None) -> str:
        selected_mode = mode if mode in ANALYSIS_MODE_CONFIG else "standard"
        selected_interval = ANALYSIS_MODE_CONFIG[selected_mode]["pose_interval"]
        if selected_mode != self.analysis_mode or selected_interval != self.pose_interval:
            self.analysis_mode = selected_mode
            self.pose_interval = selected_interval
            self._cached_pose_keypoints = []
            self._cached_pose_count = None
            logger.info(
                f"Analysis mode changed: {selected_mode}, pose interval={selected_interval}"
            )
        return self.analysis_mode

    def _create_dao_policy(self):
        if ThesisDaoPolicy is None:
            return None
        return ThesisDaoPolicy(
            deadline_ms=float(os.getenv("DAO_SLO_MS", "200")),
            max_server_workers=int(os.getenv("DAO_MAX_SERVER_WORKERS", "2")),
            local_pool_size=int(os.getenv("DAO_LOCAL_POOL_SIZE", "2")),
            initial_max_pending=int(os.getenv("DAO_INITIAL_MAX_PENDING", "2")),
            ema_alpha=float(os.getenv("DAO_EMA_ALPHA", "0.3")),
        )
    
    def _connect(self):
        """连接到边缘服务器（keepalive 保活 + 通道复用，减少 WiFi 抖动导致的断连）"""
        target = f'{self.edge_host}:{self.edge_port}'
        try:
            # 通道复用：目标地址一致且通道未关闭时直接复用，避免 WiFi 上重复 TCP+HTTP/2 握手
            if self.channel is not None and self._channel_target == target:
                state = grpc.ChannelConnectivity.SHUTDOWN
                try:
                    state = self.channel.get_state(try_to_connect=False)
                except Exception:
                    pass
                if state != grpc.ChannelConnectivity.SHUTDOWN:
                    # 通道仍活着：直接探测，探测失败也保留通道（gRPC 后台自动重连 TCP）
                    was_connected = self.edge_connected
                    self._probe_edge_health()
                    if self.edge_connected and not was_connected:
                        logger.info(f"✅ 复用现有 gRPC 通道恢复连接: {target}")
                    return
                # 通道已 SHUTDOWN，需要重建
                self._close_channel_quietly()

            self._close_channel_quietly()

            # gRPC keepalive：每 30s 发送一次 PING，防止 WiFi 空闲时被路由器/NAT 静默断链
            self.channel = grpc.insecure_channel(
                target,
                options=[
                    ('grpc.connect_timeout_ms', 5000),
                    ('grpc.keepalive_time_ms', int(self._keepalive_interval * 1000)),
                    ('grpc.keepalive_timeout_ms', int(self._keepalive_timeout * 1000)),
                    ('grpc.keepalive_permit_without_calls', 1),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.max_reconnect_backoff_ms', int(max(self._reconnect_interval, 1) * 1000)),
                ],
            )
            self._channel_target = target
            self.stub = pb2_grpc.InferenceServiceStub(self.channel)
            self._probe_edge_health()
        except Exception as e:
            logger.warning(f"⚠️ 无法创建连接: {e}")
            self._cleanup_connection()

    def _probe_edge_health(self) -> bool:
        """真实健康探测：调用 GetNodeStatus 验证服务器可达。成功置为已连接。"""
        if self.stub is None:
            return False
        try:
            response = self.stub.GetNodeStatus(pb2.StatusRequest(), timeout=self._connect_probe_timeout)
            self.edge_connected = True
            self._consecutive_offload_failures = 0
            logger.info(f"✅ 成功连接边缘服务器: {self.edge_host}:{self.edge_port}")
            logger.info(f"   节点ID: {response.node_id}")
            logger.info(f"   队列长度: {response.queue_length}")
            logger.info(f"   平均延迟: {response.avg_latency_ms:.1f}ms")
            return True
        except grpc.FutureTimeoutError:
            logger.warning(f"⚠️ 连接边缘服务器超时({self._connect_probe_timeout}s): {self.edge_host}:{self.edge_port}")
        except grpc.RpcError as e:
            logger.warning(f"⚠️ 边缘服务器响应错误: {e.code()} - {e.details()}")
        except Exception as e:
            logger.warning(f"⚠️ 边缘服务器探测异常: {e}")
        # 探测失败仅标记断连，保留 channel 供下次复用（gRPC 后台自动重连）
        self.edge_connected = False
        return False

    def _close_channel_quietly(self):
        """安全关闭旧 gRPC 通道，避免句柄泄漏。"""
        channel, self.channel, self.stub = self.channel, None, None
        self._channel_target = None
        self.edge_connected = False
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass

    def _cleanup_connection(self):
        """完全清理连接（关闭通道）。"""
        self._close_channel_quietly()
        logger.info("切换到本地推理模式")
    
    def _init_local_models(self):
        """初始化本地模型"""
        if os.getenv("EDGE_FORCE_LOCAL_MODELS", "").lower() not in {"1", "true", "yes", "on"}:
            machine = platform.machine().lower()
            if platform.system().lower() == "linux" and any(tag in machine for tag in {"arm", "aarch64", "armv7"}):
                logger.info("轻客户端模式：树莓派默认不加载本地 ultralytics 模型，仅使用边缘卸载。")
                self.yolo_model = None
                self.pose_model = None
                self.pipeline = None
                return
        try:
            from ultralytics import YOLO
            from src.inference.education_pipeline import EducationVisionPipeline
            import onnxruntime as ort
            
            # YOLO检测模型
            base_dir = Path(__file__).parent
            models_dir = base_dir / "data" / "models"
            yolo_candidates = [
                base_dir / "yolov8n.pt",
                models_dir / "yolov8n.pt",
                base_dir / "yolov8n.onnx",
                models_dir / "yolov8n.onnx",
            ]
            yolo_path = next((path for path in yolo_candidates if path.exists()), None)
            
            if yolo_path is not None:
                self.yolo_model = YOLO(str(yolo_path))
                logger.info(f"YOLO模型加载成功: {yolo_path}")
            else:
                logger.warning("未找到 YOLO 检测模型")
            
            # 姿态模型
            try:
                pose_path = base_dir / "yolov8n-pose.pt"
                self.pose_model = YOLO(str(pose_path) if pose_path.exists() else "yolov8n-pose.pt")
                logger.info("姿态模型加载成功")
            except:
                logger.warning("姿态模型加载失败")
            
            # 教育分析流水线
            self.pipeline = EducationVisionPipeline(
                ort_module=ort,
                providers=["CPUExecutionProvider"],
                face_model=None,
                expression_model=None,
                head_pose_model=None,
                gaze_model=None,
                learning_state_model=None,
            )
            logger.info("教育分析流水线初始化成功")
        except Exception as e:
            logger.error(f"本地模型初始化失败: {e}")
    
    def _yolo_detect(self, frame):
        """YOLO人物检测"""
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model(frame, conf=0.12, classes=[0, 67, 73], verbose=False)
        detections = []
        object_detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                if class_id == 0 and confidence < 0.25:
                    continue
                if class_id == 67 and confidence < 0.12:
                    continue
                if class_id == 73 and confidence < 0.20:
                    continue
                detection = {
                    "x1": max(0, x1), "y1": max(0, y1),
                    "x2": min(w-1, x2), "y2": min(h-1, y2),
                    "confidence": confidence, "class_id": class_id
                }
                if class_id == 0:
                    detections.append(detection)
                else:
                    object_detections.append(detection)

        self._last_object_detections = object_detections
        
        return detections
    
    def _pose_estimate(self, frame, detections):
        """姿态估计"""
        if self.pose_model is None:
            return []
        
        pose_keypoints = []
        h, w = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            if x2 <= x1 or y2 <= y1:
                pose_keypoints.append([])
                continue
            
            margin = 30
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(w, x2 + margin)
            y2_crop = min(h, y2 + margin)
            
            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            if person_crop.size == 0:
                pose_keypoints.append([])
                continue
            
            results = self.pose_model(person_crop, verbose=False)
            keypoints = []
            for result in results:
                if (
                    result.keypoints is not None
                    and getattr(result.keypoints, "xy", None) is not None
                    and len(result.keypoints.xy) > 0
                ):
                    xy = result.keypoints.xy[0]
                    conf = None
                    if getattr(result.keypoints, "conf", None) is not None:
                        try:
                            conf = result.keypoints.conf[0]
                        except (IndexError, TypeError):
                            pass
                    for i, kp in enumerate(xy):
                        c = float(conf[i]) if conf is not None and i < len(conf) else 1.0
                        keypoints.append([float(kp[0]) + x1_crop, float(kp[1]) + y1_crop, c])
                    break
            
            pose_keypoints.append(keypoints if keypoints else [])

        return pose_keypoints

    @staticmethod
    def _normalize_utilization(value):
        value = float(value or 0.0)
        if value > 1.0:
            value /= 100.0
        return max(0.0, min(value, 1.0))

    def _build_realtime_dag(self):
        if build_education_pipeline is None:
            return None
        arrival_time = time.time() * 1000.0
        return build_education_pipeline(
            dag_id=f"edu_live_{getattr(self, '_frame_count', 0)}",
            deadline_ms=float(os.getenv("DAO_SLO_MS", "200")),
            arrival_time=arrival_time,
            density_level=os.getenv("DAO_DENSITY_LEVEL", "medium"),
        )

    def _run_dao_decision(self, status=None, local_cpu=0.0, local_score=0.0):
        """使用论文 DaoPolicy (Algorithm 4.1) 做帧级卸载决策。

        DaoPolicy 通过内部 EMA 延迟估计 + 三层级联 + AIMD 直接返回 bool，
        无需构建 DAG 对象。status/local_cpu/local_score 仅用于前端面板展示。
        """
        if self.dao_policy is None:
            return None

        force_offload = (
            self.edge_connected
            and getattr(self, "_frame_count", 0) <= self._force_offload_frames
        )
        should_offload = True if force_offload else self.dao_policy.decide()
        # 轻客户端保护：本机未加载任何本地模型（树莓派轻量部署）时，本地分支是
        # 空跑——无检测结果（学生端会被判缺勤），且空跑耗时会把 local EMA 拉低
        # 到 DAO 永远选中本地。只要边缘可达，就不允许把帧留在本地。
        lightweight_force = (
            not should_offload
            and self.edge_connected
            and self.yolo_model is None
            and self.pose_model is None
            and self.pipeline is None
        )
        if lightweight_force:
            should_offload = True
        strategy = "offload" if should_offload else "local"

        # 读取 DaoPolicy 内部状态用于展示
        with self.dao_policy._lock:
            pending_offload = self.dao_policy._pending_offload
            pending_local = self.dao_policy._pending_local
            max_pending = self.dao_policy._max_pending_offload
            aimd_accum = self.dao_policy._aimd_accum
            offload_ema = self.dao_policy._offload_ema
            local_ema = self.dao_policy._local_ema

        # 判定三层级联中哪个层级触发了决策
        if pending_offload >= max_pending:
            gate = "hard_gate"
            reason = f"Hard Gate: 卸载管道已满 ({pending_offload}/{max_pending})，强制本地。"
        elif pending_local >= self.dao_policy._local_pool_size:
            gate = "load_balance_gate"
            reason = f"负载均衡 Gate: 本地管道已满 ({pending_local}/{self.dao_policy._local_pool_size})，强制卸载。"
        else:
            gate = "soft_decision"
            offload_cost = offload_ema * (1.0 + pending_offload / max(self.dao_policy._max_server_workers, 1))
            local_cost = local_ema * (1.0 + pending_local / max(self.dao_policy._local_pool_size, 1))
            reason = (
                f"Soft Decision: 卸载成本={offload_cost:.1f}ms vs 本地成本={local_cost:.1f}ms，"
                f"选择{'卸载' if should_offload else '本地'}。"
            )
        if force_offload:
            gate = "warmup_force_offload"
            reason = f"DAO 预热：边缘服务器已连接，前 {self._force_offload_frames} 帧强制卸载以建立真实延迟 EMA 与管理端任务流。"
        elif lightweight_force:
            gate = "lightweight_force_offload"
            reason = "轻客户端保护：本机未加载本地模型，本地执行无法检测（会被判缺勤）且空跑耗时污染本地 EMA；边缘服务器可达，强制卸载。"

        edge_queue = int(status.queue_length) if status else 0
        edge_latency = float(status.avg_latency_ms or 0.0) if status else 0.0
        edge_cpu = self._normalize_utilization(status.cpu_utilization) if status else 0.0
        edge_gpu = self._normalize_utilization(status.gpu_utilization) if status else 0.0

        self.dao_decision_info = {
            "enabled": True,
            "policy": "DaoPolicy",
            "strategy": strategy,
            "reason": reason,
            "admission": gate,
            "priority": 0.0,
            "stage_count": 5,
            "target_node_id": f"{self.edge_host}:{self.edge_port}" if should_offload else None,
            "stages_to_offload": ["YOLO Detect", "YOLO-pose", "State Fusion", "Response"] if should_offload else [],
            "recent_offloads": pending_offload,
            "congestion_window_ms": None,
            "max_offloads_per_window": max_pending,
            "estimated_transfer_ms": None,
            "deadline_ms": self.dao_policy._deadline_ms,
            "density_level": None,
            "is_gpu_heavy": None,
            "neighbors": [],
            "edge_queue": edge_queue,
            "edge_latency_ms": edge_latency,
            "edge_cpu": edge_cpu,
            "edge_gpu": edge_gpu,
            "local_cpu": float(local_cpu),
            "local_score": float(local_score),
            "pending_offload": pending_offload,
            "pending_local": pending_local,
            "max_pending_offload": max_pending,
            "aimd_accum": round(aimd_accum, 2),
            "offload_ema": round(offload_ema, 1),
            "local_ema": round(local_ema, 1),
            "execution_note": "论文 DAO Algorithm 4.1: 三层级联决策 + AIMD 管道宽度控制。",
        }
        return strategy

    def _refresh_edge_status(self):
        """周期性查询边缘状态，仅用于前端面板展示。"""
        if not self.edge_connected or self.stub is None:
            return
        try:
            status = self.stub.GetNodeStatus(pb2.StatusRequest(), timeout=self._connect_probe_timeout)
            local_cpu = DeviceProfiler.get_cpu_percent(interval=0.1)
            local_score = DeviceProfiler.estimate_performance_score()
            self._run_dao_decision(status, local_cpu, local_score)
        except Exception as e:
            logger.warning(f"[DAO] 边缘状态刷新失败: {e}")
    
    def _select_strategy(self):
        """DAO 不可用时的兜底策略选择。"""
        local_cpu = DeviceProfiler.get_cpu_percent(interval=0.5)
        local_score = DeviceProfiler.estimate_performance_score()

        logger.info(f"[DAO-fallback] 本地分数: {local_score:.1f}/100, CPU: {local_cpu:.1f}%")

        if not self.edge_connected:
            self.dao_decision_info = {
                **self.dao_decision_info,
                "strategy": "local",
                "admission": "local_no_edge",
                "reason": "无可用边缘服务器。",
                "local_cpu": float(local_cpu),
                "local_score": float(local_score),
            }
            return 'local'

        status = None
        if self.stub is not None:
            try:
                status = self.stub.GetNodeStatus(pb2.StatusRequest(), timeout=self._connect_probe_timeout)
            except Exception as e:
                logger.warning(f"[DAO-fallback] 边缘状态查询失败: {e}")

        if local_score > 50 and local_cpu < 60:
            strategy = 'local'
            reason = "DAO不可用，本地性能充足。"
        elif status and status.queue_length < 3 and status.avg_latency_ms < 100:
            strategy = 'offload'
            reason = "DAO不可用，边缘负载低。"
        else:
            strategy = 'local'
            reason = "DAO不可用，默认本地。"
        self.dao_decision_info = {
            **self.dao_decision_info,
            "enabled": False,
            "policy": "fallback-rule",
            "strategy": strategy,
            "admission": "fallback",
            "reason": reason,
            "edge_queue": int(status.queue_length) if status else 0,
            "edge_latency_ms": float(status.avg_latency_ms or 0.0) if status else 0.0,
            "local_cpu": float(local_cpu),
            "local_score": float(local_score),
        }
        return strategy
    
    def analyze_frame(self, frame, analysis_mode: str | None = None, device_id: str | None = None):
        """分析单帧 — 使用论文 DAO (Algorithm 4.1) 逐帧决策。"""
        self.set_analysis_mode(analysis_mode)
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            self._last_status_check = 0

        self._frame_count += 1
        # 断连后按时间节流自动重连（不依赖帧率），避免需要手动 curl 恢复
        _now = time.time()
        if (
            not self.edge_connected
            and self.edge_host
            and self.edge_port
            and _now - self._last_reconnect_ts >= self._reconnect_interval
        ):
            self._last_reconnect_ts = _now
            self._connect()

        # ---- 逐帧 DAO 决策 (O(1)) ----
        use_dao = self.dao_policy is not None and self.edge_connected
        if use_dao:
            strategy = self._run_dao_decision(
                local_cpu=DeviceProfiler.get_cpu_percent(interval=0.0),
                local_score=DeviceProfiler.estimate_performance_score(),
            )
            should_offload = strategy == "offload"
            if should_offload:
                self.dao_decision_info = {
                **self.dao_decision_info,
                "strategy": "offload",
                "admission": "accept",
                "reason": "边缘服务器已连接，课堂视频帧进入 Windows 边缘服务器执行 YOLO + 姿态估计 + 状态融合推理，便于 DAO 管理端展示真实任务队列与甘特图。",
                "target_node_id": f"{self.edge_host}:{self.edge_port}",
                "stages_to_offload": ["YOLO Detect", "YOLO-pose", "State Fusion", "Response"],
            }
            self.strategy = 'offload' if should_offload else 'local'
        else:
            # 无 DaoPolicy 时：每 30 帧用兜底规则重新评估
            if self._frame_count == 1 or self._frame_count - self._last_status_check > 30:
                self.strategy = self._select_strategy()
                self._last_status_check = self._frame_count

        # 周期性刷新边缘状态（仅前端面板展示用）
        if self._frame_count - self._last_status_check > 30:
            if use_dao:
                self._refresh_edge_status()
            self._last_status_check = self._frame_count

        # ---- 执行分析 + DaoPolicy 队列管理 + 结果反馈 ----
        result = None
        if self.strategy == 'offload':
            if use_dao:
                self.dao_policy.enter_offload_queue()
            result = self._offload_analyze(frame, device_id=device_id)
            if result and use_dao:
                total_ms = result.get('total_ms', 0)
                if 'tracks' in result:
                    # 实际完成卸载 → AIMD + EMA 反馈
                    self.dao_policy.report_offload_result(total_ms)
                else:
                    # 卸载失败回退本地
                    self.dao_policy.cancel_offload()
                    self.dao_policy.enter_local_queue()
                    self.dao_policy.report_local_result(total_ms)
        else:
            if use_dao:
                self.dao_policy.enter_local_queue()
            result = self._local_analyze(frame)
            if result and use_dao:
                self.dao_policy.report_local_result(result.get('total_ms', 0))

        return result
    
    def _local_analyze(self, frame):
        """本地分析"""
        start = time.perf_counter()
        detections = self._yolo_detect(frame)
        object_detections = getattr(self, "_last_object_detections", [])
        should_run_pose = (
            self._frame_count == 1
            or (self._frame_count - 1) % self.pose_interval == 0
            or self._cached_pose_count != len(detections)
        )
        if should_run_pose:
            pose_keypoints = self._pose_estimate(frame, detections)
            self._cached_pose_keypoints = pose_keypoints
            self._cached_pose_count = len(detections)
        else:
            pose_keypoints = self._cached_pose_keypoints
        
        result = None
        if self.pipeline and detections:
            result = self.pipeline.analyze_smart(frame, detections, object_detections, pose_keypoints)
        total_ms = (time.perf_counter() - start) * 1000
        
        return {
            'detections': detections,
            'object_detections': object_detections,
            'pose_keypoints': pose_keypoints,
            'analysis_result': result,
            'strategy': self.strategy,
            'total_ms': total_ms,
            'pose_refreshed': should_run_pose,
            'pose_interval': self.pose_interval,
            'analysis_mode': self.analysis_mode,
            'dao_decision': self.dao_decision_info,
        }
    
    def _offload_analyze(self, frame, device_id: str | None = None):
        """卸载到边缘分析"""
        if not self.edge_connected:
            return self._local_analyze(frame)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, jpeg_data = cv2.imencode('.jpg', frame, encode_param)
        
        request = pb2.PipelineRequest(
            device_id=device_id or f'client_{platform.node()}',
            frame_index=int(getattr(self, "_frame_count", 0)),
            dag_id='education_pipeline',
            deadline_ms=100.0,
            jpeg_frame=jpeg_data.tobytes()
        )
        
        try:
            response = self.stub.RunPipelineA(
                request,
                timeout=self._rpc_timeout,
            )
            education_summary = None
            education_json = getattr(response, "education_json", "")
            if education_json:
                try:
                    education_summary = json.loads(education_json)
                except json.JSONDecodeError as exc:
                    logger.warning(f"杈圭紭鏁欒偛鍒嗘瀽 JSON 瑙ｆ瀽澶辫触: {exc}")
            self._consecutive_offload_failures = 0
            return {
                'detections': [{'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2, 'confidence': d.confidence, 'class_id': d.class_id} for d in response.detections],
                'tracks': [{'track_id': t.track_id, 'x1': t.x1, 'y1': t.y1, 'x2': t.x2, 'y2': t.y2} for t in response.tracks],
                'analysis_result': None,
                'analysis_summary': education_summary,
                'strategy': 'offload',
                'total_ms': response.total_ms,
                'pose_refreshed': True,
                'pose_interval': self.pose_interval,
                'analysis_mode': self.analysis_mode,
                'dao_decision': self.dao_decision_info,
            }
        except Exception as e:
            self._consecutive_offload_failures += 1
            code = None
            try:
                code = e.code()
            except Exception:
                pass
            is_transient = code in (
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            )
            if (
                self._consecutive_offload_failures >= self._offload_fail_threshold
                or not is_transient
            ):
                if is_transient:
                    # 瞬时故障（超时/暂时不可达）：保留 channel 供快速恢复，仅标记断连
                    self.edge_connected = False
                    logger.warning(
                        f"卸载连续失败 {self._consecutive_offload_failures} 次 (code={code})，临时切换本地（保留通道待恢复）: {e}"
                    )
                else:
                    logger.warning(
                        f"卸载连续失败 {self._consecutive_offload_failures} 次 (code={code})，断开连接并切换本地: {e}"
                    )
                    self._cleanup_connection()
            else:
                logger.warning(
                    f"卸载瞬时失败 {self._consecutive_offload_failures}/{self._offload_fail_threshold} (code={code})，临时回退本地: {e}"
                )
            return self._local_analyze(frame)
    
    def draw_results(self, frame, result):
        """绘制结果"""
        if not result:
            return frame
        
        analysis = result.get('analysis_result')
        detections = result.get('detections', [])
        strategy = result.get('strategy', 'unknown')
        total_ms = result.get('total_ms', 0)
        
        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 如果有分析结果，绘制详细状态
        if analysis:
            summary = analysis.summary if hasattr(analysis, 'summary') else ''
            _s = max(0.4, min(0.85, frame.shape[1] / 1000.0))
            _bar_h = max(36, int(_s * 70))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], _bar_h), (0, 0, 0), -1)
            _tk = max(1, int(round(_s * 2.5)))
            (_tw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, _s, _tk)
            if _tw > frame.shape[1] - 24:
                summary = summary[:max(1, int(len(summary) * (frame.shape[1] - 24) / max(_tw, 1)) - 1)] + ".."
            cv2.putText(frame, summary, (12, int(_bar_h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, _s, (255, 255, 255), _tk, cv2.LINE_AA)
            
            students = analysis.students if hasattr(analysis, 'students') else []
            for i, student in enumerate(students):
                if i >= len(detections):
                    continue
                det = detections[i]
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                
                state = student.learning_state.label if hasattr(student, 'learning_state') else 'unknown'
                color = (0, 255, 0)
                if state == 'distracted':
                    color = (0, 0, 255)
                elif state in ['confused', 'tired']:
                    color = (0, 255, 255)
                
                _ws = frame.shape[1]
                _ls = max(0.4, min(0.85, _ws / 1000.0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(_ls * 4)))
                _tag_h = max(20, int(_ls * 34))
                _tag_y1 = max(0, y1 - _tag_h)
                cv2.rectangle(frame, (x1, _tag_y1), (min(x2, _ws), y1), color, -1)
                label = f"ID{student.track_id}: {state}" if hasattr(student, 'track_id') else f"ID{i}: {state}"
                (_lw, _lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _ls * 0.72, max(1, int(_ls * 2)))
                # 标签靠近右边界时整体左移，避免溢出视频边界
                _lx = x1 + 5
                if _lx + _lw > _ws - 4:
                    _lx = max(0, _ws - _lw - 4)
                cv2.putText(frame, label, (_lx, max(_lh + 2, _tag_y1 + _tag_h // 2 + _lh // 2)), cv2.FONT_HERSHEY_SIMPLEX, _ls * 0.72, (255,255,255), max(1, int(_ls * 2)), cv2.LINE_AA)
                
                intervention = student.learning_state.intervention if hasattr(student, 'learning_state') else ''
                if intervention and intervention != 'none':
                    cv2.putText(frame, f"-> {intervention}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制信息栏
        _ws = frame.shape[1]
        _hs = frame.shape[0]
        _is = max(0.4, min(0.85, _ws / 1000.0))
        _ibar = max(32, int(_is * 56))
        info = f"Strategy: {strategy} | Time: {total_ms:.0f}ms | Detections: {len(detections)}"
        _itk = max(1, int(round(_is * 2.5)))
        (_itw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, _is * 0.82, _itk)
        # 信息文字超出帧宽时截断，避免越过视频右边界
        if _itw > _ws - 24:
            info = info[:max(1, int(len(info) * (_ws - 24) / max(_itw, 1)) - 1)] + ".."
        cv2.rectangle(frame, (0, _hs - _ibar), (_ws, _hs), (0, 0, 0), -1)
        cv2.putText(frame, info, (12, _hs - _ibar // 2 + max(2, int(_is * 6))), cv2.FONT_HERSHEY_SIMPLEX, _is * 0.82, (0, 255, 255), _itk, cv2.LINE_AA)
        
        return frame
    
    def analyze_video(self, video_path):
        """分析视频"""
        logger.info(f"分析视频: {video_path}")
        DeviceProfiler.print_profile()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result = self.analyze_frame(frame)
            frame = self.draw_results(frame, result)
            
            cv2.imshow('Smart Education', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                logger.info("退出")
                break
            
            time.sleep(1/fps)
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"完成: {frame_count}帧")
    
    def analyze_webcam(self, camera_id=0):
        """实时摄像头分析"""
        logger.info(f"摄像头分析: {camera_id}")
        DeviceProfiler.print_profile()
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"无法打开摄像头")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.analyze_frame(frame)
            frame = self.draw_results(frame, result)
            
            cv2.imshow('Smart Education', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                logger.info("退出")
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Smart Education - Edge Offload')
    parser.add_argument('--host', default='192.168.10.22', help='Edge server IP')
    parser.add_argument('--port', type=int, default=50051, help='Edge server port')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
    parser.add_argument('--local-only', action='store_true', help='Force local only')
    
    args = parser.parse_args()
    
    client = SmartEducationOffload(args.host, args.port)
    
    if args.local_only:
        client.strategy = 'local'
    
    if args.video:
        client.analyze_video(args.video)
    elif args.webcam:
        client.analyze_webcam(args.camera_id)
    else:
        client.analyze_webcam(args.camera_id)


if __name__ == '__main__':
    main()
