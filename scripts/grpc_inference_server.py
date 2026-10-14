#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""电脑边缘服务器：gRPC 推理服务。

接收来自树莓派、安卓手机等边缘设备的卸载请求，用 ONNX Runtime 执行推理，返回结果。

用法:
    cd experiment
    .venv/bin/python scripts/grpc_inference_server.py
    .venv/bin/python scripts/grpc_inference_server.py --port 50051 --device gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
import tempfile
from collections import deque
from concurrent import futures
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np
from loguru import logger

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
YOLO_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "edge-video-ultralytics")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

import grpc
from src.grpc_gen import inference_service_pb2 as pb2
from src.grpc_gen import inference_service_pb2_grpc as pb2_grpc
from src.inference.education_pipeline import EducationVisionPipeline
from src.runtime.discovery import RuntimeAnnouncer


class InferenceServiceImpl(pb2_grpc.InferenceServiceServicer):
    """gRPC 推理服务实现。"""

    def __init__(self, yolo_model: str, osnet_model: str | None, device: str = "gpu",
                 extra_latency_ms: float = 0.0, max_workers: int = 4,
                 movenet_model: str = "yolov8n-pose.pt",
                 torch_device: str = "auto",
                 enable_reid: bool = False,
                 enable_education: bool = False,
                 face_model: str | None = None,
                 expression_model: str | None = None,
                 head_pose_model: str | None = None,
                 gaze_model: str | None = None,
                 education_csv: str | None = None,
                 face_landmarker_model: str | None = None):
        import onnxruntime as ort

        self._ort = ort
        self._device = device
        self._max_workers = max_workers
        self._torch_device = self._resolve_torch_device(torch_device)
        self._dao_slo_ms = float(os.getenv("DAO_SLO_MS", "200"))
        self._dao_congestion_window_ms = float(os.getenv("DAO_CONGESTION_WINDOW_MS", "500"))
        self._dao_max_offloads_per_window = int(os.getenv("DAO_MAX_OFFLOADS_PER_WINDOW", "3"))
        self._dao_bandwidth_mbps = float(os.getenv("DAO_NETWORK_BANDWIDTH_MBPS", "100"))
        self._dao_utilization_threshold = float(os.getenv("DAO_UTILIZATION_THRESHOLD", "0.85"))
        self._dao_bottleneck_ratio = float(os.getenv("DAO_BOTTLENECK_RATIO", "0.3"))

        yolo_path = os.path.abspath(yolo_model)
        movenet_path = os.path.abspath(movenet_model)
        self._use_ultralytics_yolo = yolo_path.lower().endswith(".pt")
        self._use_ultralytics_pose = movenet_path.lower().endswith(".pt")

        # GPU sessions — 自动检测平台可用的 Execution Provider
        # 注意: CoreML = Apple Mac, CUDA = NVIDIA, Dml = DirectML (通用 GPU)
        if device == "gpu":
            available_providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime 已安装的所有 EP: {available_providers}")
            gpu_providers = []
            for ep in ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]:
                if ep in available_providers:
                    gpu_providers.append(ep)
                    logger.info(f"ONNX Runtime 检测到可用 GPU EP: {ep}")
            if not gpu_providers:
                logger.warning(
                    "--device gpu 指定但未检测到 GPU EP (CUDA/Dml/CoreML)，ONNX 模型将回退 CPU。\n"
                    "  · 若仅需本地模拟/测试，CPU 推理完全够用，可忽略此提示。\n"
                    "  · 若需 GPU 加速，请安装 onnxruntime-gpu:\n"
                    "      pip uninstall onnxruntime && pip install onnxruntime-gpu"
                )
            providers = gpu_providers + ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info(f"加载视觉模型 (最终 providers={providers})...")
        self._yolo = None
        self._pose_model = None
        if self._use_ultralytics_yolo or self._use_ultralytics_pose:
            from ultralytics import YOLO
            if self._use_ultralytics_yolo:
                self._yolo = YOLO(yolo_path)
                logger.info(f"YOLO torch device: {self._torch_device}")
                logger.info(f"YOLO 检测模型使用 Ultralytics: {yolo_path}")
            if self._use_ultralytics_pose:
                self._pose_model = YOLO(movenet_path)
                logger.info(f"YOLO-pose torch device: {self._torch_device}")
                logger.info(f"YOLO-pose 姿态模型使用 Ultralytics: {movenet_path}")
        if not self._use_ultralytics_yolo:
            self._yolo = ort.InferenceSession(yolo_path, providers=providers)
            logger.info(f"YOLOv8n ONNX providers: {self._yolo.get_providers()}")
        self._osnet = None
        if enable_reid and osnet_model:
            osnet_path = os.path.abspath(osnet_model)
            self._osnet = ort.InferenceSession(osnet_path, providers=providers)
        elif enable_reid:
            logger.warning("ReID enabled but --osnet-model is empty; ReID will be skipped")
            enable_reid = False
        self._movenet = None
        if not self._use_ultralytics_pose:
            # MoveNet 用 CPU (轻量模型，避免 CoreML 编译开销)
            self._movenet = ort.InferenceSession(movenet_path, providers=["CPUExecutionProvider"])
            logger.info(f"MoveNet providers: {self._movenet.get_providers()}")
        self._enable_reid = enable_reid and self._osnet is not None
        if self._enable_reid:
            logger.info(f"OSNet providers: {self._osnet.get_providers()}")
        else:
            logger.info("OSNet/ReID disabled")

        # ── GPU 诊断总结 ──
        logger.info("=" * 60)
        logger.info("GPU 诊断总结:")
        logger.info(f"  torch device = {self._torch_device}")
        if self._torch_device.startswith("cuda"):
            try:
                import torch
                logger.info(f"  torch.cuda.get_device_name() = {torch.cuda.get_device_name(0)}")
                mem = torch.cuda.mem_get_info(0)
                logger.info(f"  GPU 显存: {mem[1] / 1024**3:.1f} GB 总量, {mem[0] / 1024**3:.1f} GB 可用")
            except Exception:
                pass
        logger.info(f"  ONNX EP (实际) = {providers}")
        if self._yolo and hasattr(self._yolo, 'get_providers'):
            logger.info(f"  YOLO ONNX EP = {self._yolo.get_providers()}")
        if self._osnet:
            logger.info(f"  OSNet ONNX EP = {self._osnet.get_providers()}")
        logger.info(f"  Education use_gpu = {device == 'gpu'}")
        logger.info("=" * 60)

        self._education = None
        self._latest_education: dict | None = None
        self._education_cache: dict[str, dict] = {}
        self._education_cache_lock = threading.Lock()
        self._education_interval = max(
            1, int(os.getenv("EDUCATION_STATE_INTERVAL", "2"))
        )
        self._education_csv_path = education_csv
        self._education_csv_file = None
        self._education_csv_writer = None
        self._education_csv_lock = threading.Lock()
        if enable_education:
            openvino_models = [face_model, expression_model, head_pose_model, gaze_model]
            missing_weights = [
                str(path) for path in openvino_models
                if path and Path(path).suffix.lower() == ".xml" and not Path(path).with_suffix(".bin").is_file()
            ]
            if missing_weights:
                logger.warning(
                    "OpenVINO .bin 权重未随提交提供，课堂服务降级为目标检测模式: {}",
                    ", ".join(missing_weights),
                )
                enable_education = False
        if enable_education:
            os.environ["DAISEE_LSTM_DEVICE"] = (
                "cuda" if self._torch_device.startswith("cuda") else "cpu"
            )
            self._education = EducationVisionPipeline(
                ort,
                providers,
                use_gpu=(device == "gpu"),
                face_model=face_model,
                expression_model=expression_model,
                head_pose_model=head_pose_model,
                gaze_model=gaze_model,
                face_landmarker_model=face_landmarker_model,
            )
            if self._education_csv_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._education_csv_path = os.path.join(
                    "data",
                    "results",
                    f"education_runtime_{stamp}.csv",
                )
            self._open_education_csv(self._education_csv_path)

        # 人工延迟：模拟弱网 / 服务器排队
        self._extra_latency_ms = extra_latency_ms
        if extra_latency_ms > 0:
            logger.info(
                f"Extra latency: {extra_latency_ms}ms per request "
                "(simulating weak network)"
            )

        # 推理信号量：控制真正并发推理数，同时测量排队等待
        self._infer_sem = threading.Semaphore(max_workers)

        # 统计
        self._lock = threading.Lock()
        self._active_requests = 0
        self._pending_requests = 0  # 等待推理的请求数（含排队中）
        self._recent_latencies: deque[float] = deque(maxlen=50)
        self._total_requests = 0
        self._stage_offload = {
            "DETECT": 0,
            "TRACK": 0,
            "REID": 0,
            "POST": 0,
        }
        self._request_sequence = 0
        self._pending_order: deque[str] = deque(maxlen=200)
        self._running_requests: dict[str, dict] = {}
        self._recent_requests: deque[dict] = deque(maxlen=1200)
        self._devices: dict[str, dict] = {}
        self._lane_available_at: dict[str, float] = {
            "Network Upload": 0.0,
            "CPU Decode": 0.0,
            "GPU YOLO": 0.0,
            "CPU Track": 0.0,
            "GPU Pose": 0.0,
            "GPU State": 0.0,
            "Network Response": 0.0,
        }

    @staticmethod
    def _resolve_torch_device(torch_device: str) -> str:
        selected = (torch_device or "auto").lower()
        if selected in {"cpu", "cuda:0"}:
            return selected
        if selected in {"cuda", "0"}:
            return "cuda:0"
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(
                    f"CUDA available for Ultralytics/State Fusion: {torch.cuda.get_device_name(0)}"
                )
                return "cuda:0"
        except Exception as exc:
            logger.warning(f"CUDA availability check failed, fallback to CPU: {exc}")
        return "cpu"

    def _sample_host_resources_unlocked(self) -> dict:
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_gb = 0.0
        memory_total_gb = 0.0
        try:
            import psutil
            cpu_percent = float(psutil.cpu_percent(interval=None))
            memory = psutil.virtual_memory()
            memory_percent = float(memory.percent)
            memory_used_gb = float(memory.used) / (1024 ** 3)
            memory_total_gb = float(memory.total) / (1024 ** 3)
        except Exception:
            cpu_percent = min(self._active_requests * 100.0 / max(self._max_workers, 1), 100.0)

        gpu_util = min(self._active_requests * 100.0 / max(self._max_workers, 1), 100.0)
        gpu_memory_percent = 0.0
        gpu_memory_used_mb = 0.0
        gpu_memory_total_mb = 0.0
        gpu_name = "CUDA/ONNX"
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.8,
                check=False,
            )
            first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if first_line:
                parts = [part.strip() for part in first_line.split(",")]
                if len(parts) >= 4:
                    gpu_name = parts[0]
                    gpu_util = float(parts[1])
                    gpu_memory_used_mb = float(parts[2])
                    gpu_memory_total_mb = float(parts[3])
                    if gpu_memory_total_mb > 0:
                        gpu_memory_percent = gpu_memory_used_mb / gpu_memory_total_mb * 100.0
        except Exception:
            pass

        idle_workers = max(0, self._max_workers - self._active_requests)
        queue_waiting = max(self._pending_requests - self._active_requests, 0)
        bottleneck = max(
            cpu_percent / 100.0,
            gpu_util / 100.0,
            memory_percent / 100.0,
            queue_waiting / max(self._max_workers, 1),
        )
        available = idle_workers > 0 and bottleneck < self._dao_utilization_threshold
        if not available and queue_waiting > 0:
            resource_state = "拥塞/排队"
        elif not available:
            resource_state = "高负载"
        elif self._active_requests == 0:
            resource_state = "空闲"
        else:
            resource_state = "可接收"

        return {
            "cpu_util": round(cpu_percent, 1),
            "memory_util": round(memory_percent, 1),
            "memory_used_gb": round(memory_used_gb, 2),
            "memory_total_gb": round(memory_total_gb, 2),
            "gpu_name": gpu_name,
            "gpu_util": round(gpu_util, 1),
            "gpu_memory_util": round(gpu_memory_percent, 1),
            "gpu_memory_used_mb": round(gpu_memory_used_mb, 1),
            "gpu_memory_total_mb": round(gpu_memory_total_mb, 1),
            "active_workers": self._active_requests,
            "idle_workers": idle_workers,
            "max_workers": self._max_workers,
            "queue_waiting": queue_waiting,
            "resource_state": resource_state,
            "available": available,
            "bottleneck": round(bottleneck, 3),
        }

    def _workflow_template(self, jpeg_bytes: int | None = None) -> list[dict]:
        upload_ms = 8.0
        if jpeg_bytes:
            upload_ms = max(4.0, jpeg_bytes * 8.0 / max(self._dao_bandwidth_mbps, 1.0) / 1000.0)
        return [
            {"name": "Frame Upload", "resource": "Network Upload", "duration_ms": round(upload_ms, 1), "desc": "用户设备上传一帧画面"},
            {"name": "Decode", "resource": "CPU Decode", "duration_ms": 10.0, "desc": "图像解码与压缩处理"},
            {"name": "YOLO Detect", "resource": "GPU YOLO", "duration_ms": 45.0, "desc": "检测人和手机等目标"},
            {"name": "Track", "resource": "CPU Track", "duration_ms": 6.0, "desc": "持续跟踪画面中的人物"},
            {"name": "YOLO-pose", "resource": "GPU Pose", "duration_ms": 35.0, "desc": "分析坐姿与肢体动作"},
            {"name": "State Fusion", "resource": "GPU State", "duration_ms": 55.0, "desc": "综合判断学习状态"},
            {"name": "Response", "resource": "Network Response", "duration_ms": 5.0, "desc": "把分析结果发回用户端"},
        ]

    def _plan_request_schedule_unlocked(self, request_id: str, arrived_at: float, jpeg_bytes: int) -> tuple[list[dict], float, float]:
        cursor = max(time.time(), arrived_at)
        schedule = []
        for step in self._workflow_template(jpeg_bytes):
            lane = step["resource"]
            start = max(cursor, self._lane_available_at.get(lane, 0.0))
            end = start + float(step["duration_ms"]) / 1000.0
            self._lane_available_at[lane] = end
            cursor = end
            schedule.append({
                "request_id": request_id,
                "stage": step["name"],
                "resource": lane,
                "planned_start": start,
                "planned_end": end,
                "duration_ms": round((end - start) * 1000.0, 1),
            })
        planned_total_ms = (schedule[-1]["planned_end"] - arrived_at) * 1000.0 if schedule else 0.0
        priority = min(1.0, max(0.05, planned_total_ms / max(self._dao_slo_ms, 1.0)))
        return schedule, planned_total_ms, priority

    def _ensure_device_unlocked(self, device_id: str, now: float | None = None) -> dict:
        now = now or time.time()
        return self._devices.setdefault(device_id, {
            "device_id": device_id,
            "first_seen": now,
            "last_seen": now,
            "total_requests": 0,
            "running": 0,
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "last_frame": None,
            "last_latency_ms": 0.0,
            "last_state": "waiting",
            "last_status": "queued",
        })

    def _touch_device(self, device_id: str, frame_index: int | None = None, status: str = "connected", state: str = "request_received") -> None:
        now = time.time()
        with self._lock:
            device = self._ensure_device_unlocked(device_id, now)
            device["last_seen"] = now
            device["last_status"] = status
            device["last_state"] = state
            if frame_index is not None:
                device["last_frame"] = int(frame_index)

    def _register_request(self, device_id: str, frame_index: int, dag_id: str, jpeg_bytes: int) -> str:
        now = time.time()
        with self._lock:
            self._request_sequence += 1
            request_id = f"task-{self._request_sequence:06d}"
            planned_schedule, planned_total_ms, dao_priority = self._plan_request_schedule_unlocked(
                request_id,
                now,
                jpeg_bytes,
            )
            admission = "accept"
            if (
                planned_total_ms > self._dao_slo_ms
                and self._pending_requests >= self._dao_max_offloads_per_window
            ):
                admission = "defer"
            item = {
                "request_id": request_id,
                "sequence": self._request_sequence,
                "device_id": device_id,
                "frame_index": int(frame_index),
                "dag_id": dag_id,
                "jpeg_bytes": int(jpeg_bytes),
                "status": "queued",
                "arrived_at": now,
                "started_at": None,
                "finished_at": None,
                "queue_wait_ms": 0.0,
                "total_ms": 0.0,
                "slo_ms": round(self._dao_slo_ms, 1),
                "slo_met": None,
                "slo_over_ms": 0.0,
                "dao_priority": round(dao_priority, 3),
                "admission": admission,
                "planned_total_ms": round(planned_total_ms, 1),
                "planned_schedule": planned_schedule,
                "stage": "waiting",
                "worker": None,
            }
            self._pending_order.append(request_id)
            self._recent_requests.appendleft(item.copy())
            device = self._ensure_device_unlocked(device_id, now)
            device["last_seen"] = now
            device["total_requests"] += 1
            device["queued"] += 1
            device["last_frame"] = int(frame_index)
            device["last_status"] = "queued"
            device["last_state"] = "等待调度"
        return request_id

    def _mark_request_running(self, request_id: str, device_id: str, queue_wait_ms: float) -> None:
        now = time.time()
        worker_name = threading.current_thread().name
        with self._lock:
            try:
                self._pending_order.remove(request_id)
            except ValueError:
                pass
            running = None
            for item in self._recent_requests:
                if item.get("request_id") == request_id:
                    item["status"] = "running"
                    item["started_at"] = now
                    item["queue_wait_ms"] = round(queue_wait_ms, 1)
                    item["stage"] = "YOLO + Pose + State Fusion"
                    item["worker"] = worker_name
                    running = item.copy()
                    break
            if running is None:
                running = {
                    "request_id": request_id,
                    "device_id": device_id,
                    "status": "running",
                    "started_at": now,
                    "queue_wait_ms": round(queue_wait_ms, 1),
                    "stage": "YOLO + Pose + State Fusion",
                    "worker": worker_name,
                }
            self._running_requests[request_id] = running
            device = self._ensure_device_unlocked(device_id, now)
            device["last_seen"] = now
            device["queued"] = max(0, int(device.get("queued", 0)) - 1)
            device["running"] = int(device.get("running", 0)) + 1
            device["last_status"] = "running"
            device["last_state"] = "执行中"

    def _finish_request(
        self,
        request_id: str,
        device_id: str,
        status: str,
        total_ms: float,
        queue_wait_ms: float,
        detections: int = 0,
        students: int = 0,
        learning_state: str = "",
        stage_timings: dict | None = None,
        actual_schedule: list[dict] | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._running_requests.pop(request_id, None)
            for item in self._recent_requests:
                if item.get("request_id") == request_id:
                    item["status"] = status
                    item["finished_at"] = now
                    item["total_ms"] = round(total_ms, 1)
                    item["queue_wait_ms"] = round(queue_wait_ms, 1)
                    item["slo_ms"] = round(self._dao_slo_ms, 1)
                    item["slo_met"] = total_ms <= self._dao_slo_ms
                    item["slo_over_ms"] = round(max(0.0, total_ms - self._dao_slo_ms), 1)
                    item["dao_priority"] = round(min(1.0, max(0.0, total_ms / max(self._dao_slo_ms, 1.0))), 3)
                    item["admission"] = "accept" if status == "completed" else status
                    item["detections"] = int(detections)
                    item["students"] = int(students)
                    item["learning_state"] = learning_state or "unknown"
                    item["stage_timings"] = stage_timings or {}
                    item["actual_schedule"] = actual_schedule or []
                    item["stage"] = "completed" if status == "completed" else status
                    break
            device = self._ensure_device_unlocked(device_id, now)
            device["last_seen"] = now
            device["running"] = max(0, int(device.get("running", 0)) - 1)
            if status == "completed":
                device["completed"] = int(device.get("completed", 0)) + 1
            elif status == "failed":
                device["failed"] = int(device.get("failed", 0)) + 1
            device["last_latency_ms"] = round(total_ms, 1)
            device["last_status"] = status
            device["last_state"] = learning_state or status

    @staticmethod
    def _education_state_label(education_summary: dict | None) -> tuple[int, str]:
        if not isinstance(education_summary, dict):
            return 0, "unknown"
        students = education_summary.get("students")
        if isinstance(students, list):
            if not students:
                return 0, "absent"
            first_state = ((students[0].get("learning_state") or {}).get("label") or "unknown")
            return len(students), str(first_state)
        learning_state = education_summary.get("learning_state")
        if isinstance(learning_state, dict):
            return int(education_summary.get("num_people_detected", 0) or 0), str(learning_state.get("label", "unknown"))
        return int(education_summary.get("num_people_detected", 0) or 0), "unknown"

    # ─── DETECT ─────────────────────────────────────────

    def _open_education_csv(self, path: str) -> None:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        self._education_csv_file = open(abs_path, "a", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp", "device_id", "frame", "dag_id", "jpeg_bytes",
            "total_ms", "detect_ms", "track_ms", "reid_ms", "pose_ms", "edu_ms",
            "queue_wait_ms", "num_detections", "num_tracks", "num_poses",
            "num_students", "analysis_mode",
            "track_id", "priority_score", "needs_attention",
            "state", "confidence", "intervention", "reasons",
            "face_present", "expression", "expression_confidence",
            "gaze_region", "head_yaw", "head_pitch", "head_roll",
            "behavior", "behavior_confidence", "behavior_evidence",
            "priority_feedback",
        ]
        self._education_csv_writer = csv.DictWriter(self._education_csv_file, fieldnames=fieldnames)
        if self._education_csv_file.tell() == 0:
            self._education_csv_writer.writeheader()
            self._education_csv_file.flush()
        logger.info(f"Education CSV enabled: {abs_path}")

    def _write_education_csv(
        self,
        *,
        device_id: str,
        frame_index: int,
        dag_id: str,
        jpeg_bytes: int,
        total_ms: float,
        detect_ms: float,
        track_ms: float,
        reid_ms: float,
        pose_ms: float,
        education_ms: float,
        queue_wait_ms: float,
        detections: list[dict],
        tracks: list[dict],
        num_poses: int,
        education_summary: dict | None,
    ) -> None:
        if self._education_csv_writer is None or self._education_csv_file is None:
            return
        summary = education_summary or {}

        # 单学生模式
        if "learning_state" in summary:
            state = summary.get("learning_state", {})
            face = summary.get("face", {})
            head_gaze = summary.get("head_gaze", {})
            behavior = summary.get("behavior", {})
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "device_id": device_id,
                "frame": frame_index,
                "dag_id": dag_id,
                "jpeg_bytes": jpeg_bytes,
                "total_ms": f"{total_ms:.3f}",
                "detect_ms": f"{detect_ms:.3f}",
                "track_ms": f"{track_ms:.3f}",
                "reid_ms": f"{reid_ms:.3f}",
                "pose_ms": f"{pose_ms:.3f}",
                "edu_ms": f"{education_ms:.3f}",
                "queue_wait_ms": f"{queue_wait_ms:.3f}",
                "num_detections": len(detections),
                "num_tracks": len(tracks),
                "num_poses": num_poses,
                "num_students": summary.get("num_people_detected", 1),
                "analysis_mode": summary.get("analysis_mode", "single"),
                "state": state.get("label", ""),
                "confidence": state.get("confidence", ""),
                "intervention": state.get("intervention", ""),
                "reasons": "|".join(state.get("reasons", [])),
                "face_present": face.get("present", ""),
                "expression": face.get("expression", ""),
                "expression_confidence": face.get("expression_confidence", ""),
                "gaze_region": head_gaze.get("gaze_region", ""),
                "head_yaw": head_gaze.get("yaw", ""),
                "head_pitch": head_gaze.get("pitch", ""),
                "head_roll": head_gaze.get("roll", ""),
                "behavior": behavior.get("label", ""),
                "behavior_confidence": behavior.get("confidence", ""),
                "behavior_evidence": "|".join(behavior.get("evidence", [])),
                "priority_feedback": "",
            }
            with self._education_csv_lock:
                self._education_csv_writer.writerow(row)
                self._education_csv_file.flush()
            return

        # 多学生模式
        students = summary.get("students", [])
        if not students:
            return

        for student in students:
            state = student.get("learning_state", {})
            face = student.get("face", {})
            head_gaze = student.get("head_gaze", {})
            behavior = student.get("behavior", {})
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "device_id": device_id,
                "frame": frame_index,
                "dag_id": dag_id,
                "jpeg_bytes": jpeg_bytes,
                "total_ms": f"{total_ms:.3f}",
                "detect_ms": f"{detect_ms:.3f}",
                "track_ms": f"{track_ms:.3f}",
                "reid_ms": f"{reid_ms:.3f}",
                "pose_ms": f"{pose_ms:.3f}",
                "edu_ms": f"{education_ms:.3f}",
                "queue_wait_ms": f"{queue_wait_ms:.3f}",
                "num_detections": len(detections),
                "num_tracks": len(tracks),
                "num_poses": num_poses,
                "num_students": summary.get("total_count", len(students)),
                "analysis_mode": "multi",
                "track_id": student.get("track_id", ""),
                "priority_score": student.get("priority_score", ""),
                "needs_attention": student.get("needs_attention", ""),
                "state": state.get("label", ""),
                "confidence": state.get("confidence", ""),
                "intervention": state.get("intervention", ""),
                "reasons": "|".join(state.get("reasons", [])),
                "face_present": face.get("present", ""),
                "expression": face.get("expression", ""),
                "expression_confidence": face.get("expression_confidence", ""),
                "gaze_region": head_gaze.get("gaze_region", ""),
                "head_yaw": head_gaze.get("yaw", ""),
                "head_pitch": head_gaze.get("pitch", ""),
                "head_roll": head_gaze.get("roll", ""),
                "behavior": behavior.get("label", ""),
                "behavior_confidence": behavior.get("confidence", ""),
                "behavior_evidence": "|".join(behavior.get("evidence", [])),
                "priority_feedback": "|".join(summary.get("priority_feedback", [])[:3]),
            }
            with self._education_csv_lock:
                self._education_csv_writer.writerow(row)
                self._education_csv_file.flush()

    def _detect(self, jpeg_bytes: bytes) -> tuple[list[dict], dict, np.ndarray, list[dict]]:
        """执行 YOLO 检测，返回 (detections, timing_dict, frame)。"""
        timing = {}

        # 解码 JPEG + resize
        t0 = time.perf_counter()
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return [], {"decode_ms": 0, "preprocess_ms": 0, "infer_ms": 0,
                        "postprocess_ms": 0}, np.zeros((640, 640, 3), dtype=np.uint8), []
        h_orig, w_orig = frame.shape[:2]
        timing["decode_ms"] = (time.perf_counter() - t0) * 1000

        if self._use_ultralytics_yolo:
            timing["preprocess_ms"] = 0.0
            t2 = time.perf_counter()
            results = self._yolo(
                frame,
                conf=0.08,
                classes=[0, 67, 73],
                imgsz=int(os.getenv("EDUCATION_YOLO_IMGSZ", "320")),
                half=self._torch_device.startswith("cuda"),
                verbose=False,
                device=self._torch_device,
            )
            timing["infer_ms"] = (time.perf_counter() - t2) * 1000
            t3 = time.perf_counter()
            detections = []
            object_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    if class_id == 0 and confidence < 0.25:
                        continue
                    if class_id == 67 and confidence < 0.08:
                        continue
                    if class_id == 73 and confidence < 0.20:
                        continue
                    item = {
                        "x1": max(0.0, x1), "y1": max(0.0, y1),
                        "x2": min(float(w_orig - 1), x2), "y2": min(float(h_orig - 1), y2),
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                    if class_id == 0:
                        detections.append(item)
                    else:
                        object_detections.append(item)
            timing["postprocess_ms"] = (time.perf_counter() - t3) * 1000
            return detections, timing, frame, object_detections

        img = cv2.resize(frame, (640, 640))

        # 预处理: RGB 转换 + 归一化 + 转置
        t1 = time.perf_counter()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        timing["preprocess_ms"] = (time.perf_counter() - t1) * 1000

        # 推理
        input_name = self._yolo.get_inputs()[0].name
        t2 = time.perf_counter()
        output = self._yolo.run(None, {input_name: img})
        timing["infer_ms"] = (time.perf_counter() - t2) * 1000

        # 后处理
        t3 = time.perf_counter()
        preds = output[0][0].T
        scores = preds[:, 4:].max(axis=1)
        mask = scores > 0.5
        boxes = preds[mask, :4]
        class_ids = preds[mask, 4:].argmax(axis=1)
        det_scores = scores[mask]

        # 只保留 person (class_id=0)
        # NMS
        if len(boxes) > 0:
            # cx,cy,w,h → x1,y1,x2,y2 (像素坐标用于 NMS)
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            nms_x1 = cx - bw / 2
            nms_y1 = cy - bh / 2
            nms_x2 = cx + bw / 2
            nms_y2 = cy + bh / 2
            nms_boxes = np.stack([nms_x1, nms_y1, nms_x2, nms_y2], axis=1)
            keep = self._nms(nms_boxes, det_scores, iou_threshold=0.5)
            boxes = boxes[keep]
            det_scores = det_scores[keep]
            class_ids = class_ids[keep]

        object_detections = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x1 = (cx - w / 2) * w_orig / 640
            y1 = (cy - h / 2) * h_orig / 640
            x2 = (cx + w / 2) * w_orig / 640
            y2 = (cy + h / 2) * h_orig / 640
            object_detections.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "confidence": float(det_scores[i]),
                "class_id": int(class_ids[i]),
            })

        person_mask = class_ids == 0
        boxes = boxes[person_mask]
        det_scores = det_scores[person_mask]

        detections = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x1 = (cx - w / 2) * w_orig / 640
            y1 = (cy - h / 2) * h_orig / 640
            x2 = (cx + w / 2) * w_orig / 640
            y2 = (cy + h / 2) * h_orig / 640
            detections.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "confidence": float(det_scores[i]),
                "class_id": 0,
            })
        timing["postprocess_ms"] = (time.perf_counter() - t3) * 1000

        return detections, timing, frame, object_detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
        """简单 NMS 实现。boxes: (N, 4) x1y1x2y2, scores: (N,)。"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    # ─── REID ───────────────────────────────────────────

    def _reid(self, frame: np.ndarray, detections: list[dict]) -> tuple[list[bytes], dict]:
        """执行 REID，返回 (embeddings_bytes, timing_dict)。"""
        timing = {"preprocess_ms": 0.0, "infer_ms": 0.0}
        if not self._enable_reid or self._osnet is None:
            return [], timing
        if not detections:
            return [], timing

        h, w = frame.shape[:2]

        # 预处理: 裁剪 + resize + 归一化
        t0 = time.perf_counter()
        crops = []
        for det in detections[:30]:
            x1 = max(0, int(det["x1"]))
            y1 = max(0, int(det["y1"]))
            x2 = min(w, int(det["x2"]))
            y2 = min(h, int(det["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = np.transpose(crop, (2, 0, 1))
            crops.append(crop)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        if not crops:
            return [], timing

        # 推理
        input_name = self._osnet.get_inputs()[0].name
        embeddings = []

        t1 = time.perf_counter()
        for crop in crops:
            out = self._osnet.run(None, {input_name: crop[np.newaxis]})
            embeddings.append(out[0].tobytes())
        timing["infer_ms"] = (time.perf_counter() - t1) * 1000

        return embeddings, timing

    # ─── POSE ───────────────────────────────────────────

    def _pose(self, frame: np.ndarray, detections: list[dict]) -> tuple[float, int, list[list[list[float]]]]:
        """执行 MoveNet 逐人姿态估计，返回 (pose_ms, num_poses)。

        与 Android MoveNetPose 逻辑一致: 裁剪每个检测框 → resize 192×192 → uint8 输入。
        """
        if not detections:
            return 0.0, 0, []

        h, w = frame.shape[:2]
        if self._use_ultralytics_pose:
            t0 = time.perf_counter()
            all_keypoints = []
            for det in detections[:50]:
                x1 = max(0, int(det["x1"]))
                y1 = max(0, int(det["y1"]))
                x2 = min(w, int(det["x2"]))
                y2 = min(h, int(det["y2"]))
                if x2 <= x1 or y2 <= y1:
                    all_keypoints.append([])
                    continue
                crop = frame[y1:y2, x1:x2]
                results = self._pose_model(
                    crop,
                    imgsz=int(os.getenv("EDUCATION_POSE_IMGSZ", "256")),
                    half=self._torch_device.startswith("cuda"),
                    verbose=False,
                    device=self._torch_device,
                )
                keypoints = []
                if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                    points = results[0].keypoints.data[0].detach().cpu().numpy()
                    for px, py, score in points[:17]:
                        keypoints.append([float(x1 + px), float(y1 + py), float(score)])
                all_keypoints.append(keypoints)
            pose_ms = (time.perf_counter() - t0) * 1000
            return pose_ms, sum(1 for item in all_keypoints if item), all_keypoints

        input_name = self._movenet.get_inputs()[0].name
        input_type = self._movenet.get_inputs()[0].type
        t0 = time.perf_counter()

        count = 0
        all_keypoints = []
        for det in detections[:50]:  # 上限与 Android 一致
            x1 = max(0, int(det["x1"]))
            y1 = max(0, int(det["y1"]))
            x2 = min(w, int(det["x2"]))
            y2 = min(h, int(det["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (192, 192))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # MoveNet 输入: uint8, NHWC
            dtype = np.int32 if "int32" in input_type else np.uint8
            inp = crop[np.newaxis].astype(dtype)
            output = self._movenet.run(None, {input_name: inp})
            raw = np.asarray(output[0]).reshape(-1, 3)
            crop_w = max(x2 - x1, 1)
            crop_h = max(y2 - y1, 1)
            keypoints = []
            for ky, kx, score in raw[:17]:
                keypoints.append([
                    float(x1 + kx * crop_w),
                    float(y1 + ky * crop_h),
                    float(score),
                ])
            all_keypoints.append(keypoints)
            count += 1

        pose_ms = (time.perf_counter() - t0) * 1000
        return pose_ms, count, all_keypoints

    # ─── gRPC 方法实现 ──────────────────────────────────

    def _apply_extra_latency(self):
        """模拟弱网延迟（在请求开始和结束各加一半）。"""
        if self._extra_latency_ms > 0:
            time.sleep(self._extra_latency_ms / 2000.0)  # 半程延迟

    def RunDetect(self, request, context):
        self._apply_extra_latency()  # 模拟上行延迟
        with self._lock:
            self._active_requests += 1

        t0 = time.perf_counter()
        detections, det_timing, _, _ = self._detect(request.jpeg_frame)
        total_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            self._active_requests -= 1
            self._recent_latencies.append(total_ms)
            self._total_requests += 1
            self._stage_offload["DETECT"] += 1

        resp = pb2.DetectResponse(
            latency_ms=det_timing["infer_ms"],
            processor=self._device,
        )
        for det in detections:
            resp.detections.append(pb2.Detection(
                x1=det["x1"], y1=det["y1"],
                x2=det["x2"], y2=det["y2"],
                confidence=det["confidence"],
                class_id=det["class_id"],
            ))
        self._apply_extra_latency()  # 模拟下行延迟
        return resp

    def RunPipelineA(self, request, context):
        t_arrive = time.perf_counter()
        device_id = request.device_id or "unknown"
        self._touch_device(device_id, request.frame_index)
        logger.info(
            f"[{device_id}] recv RunPipelineA "
            f"frame={request.frame_index} bytes={len(request.jpeg_frame)} "
            f"dag={request.dag_id}"
        )

        # 记录 pending（等待+执行中的总请求数）
        with self._lock:
            self._pending_requests += 1
            pending = self._pending_requests

        self._apply_extra_latency()  # 模拟上行延迟

        request_id = self._register_request(
            device_id=device_id,
            frame_index=request.frame_index,
            dag_id=request.dag_id,
            jpeg_bytes=len(request.jpeg_frame),
        )

        # 排队等待: 获取推理信号量（真正的瓶颈点）
        self._infer_sem.acquire()
        with self._lock:
            active = self._active_requests
            self._active_requests += 1
        queue_wait_ms = (time.perf_counter() - t_arrive) * 1000
        self._mark_request_running(request_id, device_id, queue_wait_ms)

        t_start = time.perf_counter()
        actual_schedule = []

        def record_stage(name: str, resource: str, wall_start: float, wall_end: float) -> None:
            actual_schedule.append({
                "request_id": request_id,
                "stage": name,
                "resource": resource,
                "actual_start": wall_start,
                "actual_end": wall_end,
                "duration_ms": round((wall_end - wall_start) * 1000.0, 1),
            })

        # DETECT (返回细粒度计时)
        t_wall = time.time()
        detections, det_timing, frame, object_detections = self._detect(request.jpeg_frame)
        record_stage("YOLO Detect", "GPU YOLO", t_wall, time.time())

        # TRACK (简化: 无状态, 直接用检测框作为 track)
        t_track = time.perf_counter()
        t_wall = time.time()
        tracks = []
        for i, det in enumerate(detections):
            tracks.append({
                "track_id": i,
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
            })
        track_ms = (time.perf_counter() - t_track) * 1000
        record_stage("Track", "CPU Track", t_wall, time.time())

        # REID (返回细粒度计时)
        t_wall = time.time()
        embeddings, reid_timing = self._reid(frame, detections)
        record_stage("ReID", "CPU Track", t_wall, time.time())

        # POSE (MoveNet 逐人姿态估计)
        t_wall = time.time()
        pose_ms, num_poses, pose_keypoints = self._pose(frame, detections)
        record_stage("YOLO-pose", "GPU Pose", t_wall, time.time())

        education_ms = 0.0
        education_summary = None
        smart_analysis = None
        if self._education is not None:
            t_wall = time.time()
            with self._education_cache_lock:
                cached = self._education_cache.get(device_id)
            refresh_state = (
                cached is None
                or request.frame_index % self._education_interval == 0
                or cached.get("person_count") != len(detections)
            )
            if refresh_state:
                t_edu = time.perf_counter()
                smart_analysis = self._education.analyze_smart(
                    frame,
                    detections,
                    object_detections,
                    pose_keypoints=pose_keypoints,
                )
                education_ms = (time.perf_counter() - t_edu) * 1000
                education_summary = smart_analysis.to_dict()
                with self._education_cache_lock:
                    self._education_cache[device_id] = {
                        "person_count": len(detections),
                        "summary": education_summary,
                    }
            else:
                # 学习状态在相邻视频帧中变化很慢。复用上一帧状态可省去重复的
                # 表情/视线/行为融合，同时人物检测和姿态仍逐帧真实执行。
                smart_analysis = None
                education_summary = dict(cached["summary"])
                education_summary["temporal_reused"] = True
            record_stage("State Fusion", "GPU State", t_wall, time.time())

            # 生成日志输出
            if smart_analysis is not None and smart_analysis.total_count > 0:
                logger.info(
                    f"[{device_id}] education frame={request.frame_index} "
                    f"{smart_analysis.summary}"
                )
                if smart_analysis.priority_feedback:
                    for feedback in smart_analysis.priority_feedback[:3]:  # 最多显示3条
                        logger.info(f"[{device_id}] ATTENTION: {feedback}")

            with self._lock:
                self._latest_education = {
                    "frame_index": request.frame_index,
                    "device_id": device_id,
                    **education_summary,
                }

        total_ms = (time.perf_counter() - t_start) * 1000

        # detect_ms / reid_ms 保持兼容: 返回整个 stage 的时间 (含预处理)
        detect_ms_compat = (det_timing["decode_ms"] + det_timing["preprocess_ms"]
                           + det_timing["infer_ms"] + det_timing["postprocess_ms"])
        reid_ms_compat = reid_timing["preprocess_ms"] + reid_timing["infer_ms"]
        record_stage("Response", "Network Response", time.time(), time.time() + 0.001)

        self._infer_sem.release()
        with self._lock:
            self._active_requests -= 1
            self._pending_requests -= 1
            self._recent_latencies.append(total_ms)
            self._total_requests += 1
            self._stage_offload["DETECT"] += 1
            self._stage_offload["TRACK"] += 1
            self._stage_offload["REID"] += 1
            self._stage_offload["POST"] += 1

        logger.info(f"[{device_id}] frame={request.frame_index} "
                     f"total={total_ms:.1f}ms det={detect_ms_compat:.1f}ms "
                     f"reid={reid_ms_compat:.1f}ms pose={pose_ms:.1f}ms({num_poses}p) "
                     f"edu={education_ms:.1f}ms "
                     f"qwait={queue_wait_ms:.1f}ms active={active} pending={pending}")
        self._write_education_csv(
            device_id=device_id,
            frame_index=request.frame_index,
            dag_id=request.dag_id,
            jpeg_bytes=len(request.jpeg_frame),
            total_ms=total_ms,
            detect_ms=detect_ms_compat,
            track_ms=track_ms,
            reid_ms=reid_ms_compat,
            pose_ms=pose_ms,
            education_ms=education_ms,
            queue_wait_ms=queue_wait_ms,
            detections=detections,
            tracks=tracks,
            num_poses=num_poses,
            education_summary=education_summary,
        )
        if smart_analysis is not None and smart_analysis.total_count == 0:
            logger.warning(
                f"[{device_id}] No students detected in frame={request.frame_index}"
            )
        students_count, learning_state = self._education_state_label(education_summary)
        self._finish_request(
            request_id=request_id,
            device_id=device_id,
            status="completed",
            total_ms=total_ms,
            queue_wait_ms=queue_wait_ms,
            detections=len(detections),
            students=students_count,
            learning_state=learning_state,
            stage_timings={
                "YOLO": round(detect_ms_compat, 1),
                "Track": round(track_ms, 1),
                "ReID": round(reid_ms_compat, 1),
                "Pose": round(pose_ms, 1),
                "状态融合": round(education_ms, 1),
            },
            actual_schedule=actual_schedule,
        )

        resp = pb2.PipelineResponse(
            detect_ms=detect_ms_compat,
            track_ms=track_ms,
            reid_ms=reid_ms_compat,
            pose_ms=pose_ms,
            total_ms=total_ms,
            detect_on=self._device,
            reid_on=self._device,
            pose_on="cpu",
            # 细粒度计时 (v2)
            decode_ms=det_timing["decode_ms"],
            detect_preprocess_ms=det_timing["preprocess_ms"],
            detect_infer_ms=det_timing["infer_ms"],
            detect_postprocess_ms=det_timing["postprocess_ms"],
            reid_preprocess_ms=reid_timing["preprocess_ms"],
            reid_infer_ms=reid_timing["infer_ms"],
            queue_wait_ms=queue_wait_ms,
            active_requests=active,
            education_json=(
                json.dumps(education_summary, ensure_ascii=False)
                if education_summary is not None
                else ""
            ),
        )
        for det in detections:
            resp.detections.append(pb2.Detection(
                x1=det["x1"], y1=det["y1"],
                x2=det["x2"], y2=det["y2"],
                confidence=det["confidence"],
                class_id=det["class_id"],
            ))
        for trk in tracks:
            resp.tracks.append(pb2.TrackInfo(
                track_id=trk["track_id"],
                x1=trk["x1"], y1=trk["y1"],
                x2=trk["x2"], y2=trk["y2"],
            ))
        self._apply_extra_latency()  # 模拟下行延迟
        return resp

    def RunReID(self, request, context):
        if not self._enable_reid or self._osnet is None:
            return pb2.ReIDResponse(
                embeddings=[],
                latency_ms=0.0,
                processor="disabled",
            )

        self._apply_extra_latency()
        with self._lock:
            self._active_requests += 1

        t0 = time.perf_counter()

        # 解码裁剪图
        crops = []
        for crop_bytes in request.track_crops:
            img_array = np.frombuffer(crop_bytes, dtype=np.uint8)
            crop = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if crop is not None:
                crop = cv2.resize(crop, (128, 256))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
                crop = np.transpose(crop, (2, 0, 1))
                crops.append(crop)

        embeddings = []
        input_name = self._osnet.get_inputs()[0].name
        for crop in crops:
            out = self._osnet.run(None, {input_name: crop[np.newaxis]})
            embeddings.append(out[0].tobytes())

        latency_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            self._active_requests -= 1
            self._recent_latencies.append(latency_ms)
            self._total_requests += 1
            self._stage_offload["REID"] += 1

        return pb2.ReIDResponse(
            embeddings=embeddings,
            latency_ms=latency_ms,
            processor=self._device,
        )

    def GetNodeStatus(self, request, context):
        peer = "unknown-peer"
        try:
            peer = context.peer() or peer
        except Exception:
            pass
        peer_device_id = "edge-peer_" + (
            peer.replace("ipv4:", "")
            .replace("ipv6:", "")
            .replace(":", "_")
            .replace("/", "_")
        )
        self._touch_device(peer_device_id, status="connected", state="status_probe")
        with self._lock:
            avg_lat = (
                sum(self._recent_latencies) / len(self._recent_latencies)
                if self._recent_latencies else 0.0
            )
            return pb2.StatusResponse(
                node_id="mac_m3pro",
                gpu_utilization=min(self._active_requests * 0.3, 1.0),
                cpu_utilization=min(self._active_requests * 0.2, 1.0),
                queue_length=self._active_requests,
                active_requests=self._active_requests,
                avg_latency_ms=avg_lat,
            )

    def monitor_snapshot(self) -> dict:
        """导出 Windows 边缘服务器 DAO 调度监控快照。"""
        with self._lock:
            def is_real_device(device_id: str) -> bool:
                text = str(device_id or "").strip()
                return bool(text) and not text.startswith("edge-peer_")

            def physical_device_key(device_id: str) -> str:
                text = str(device_id or "unknown").strip()
                return text.split("__", 1)[0] if "__" in text else text

            resources = self._sample_host_resources_unlocked()
            queue_waiting = resources["queue_waiting"]
            avg_latency = (
                sum(self._recent_latencies) / len(self._recent_latencies)
                if self._recent_latencies else 0.0
            )
            completed_recent = [
                item for item in self._recent_requests
                if item.get("status") == "completed" and item.get("total_ms")
            ]
            slo_met_count = sum(1 for item in completed_recent if item.get("slo_met"))
            slo_rate = (
                slo_met_count / len(completed_recent) * 100.0
                if completed_recent else 0.0
            )
            now = time.time()
            devices = []
            for device in self._devices.values():
                devices.append({
                    **device,
                    "last_seen_age_sec": round(now - float(device.get("last_seen", now)), 1),
                })
            devices.sort(key=lambda item: item.get("last_seen", 0), reverse=True)
            visible_devices = [
                device for device in devices
                if is_real_device(device.get("device_id", ""))
            ]
            realtime_user_window_sec = 15.0
            realtime_users = sum(
                1
                for device in visible_devices
                if float(device.get("last_seen_age_sec", realtime_user_window_sec + 1)) <= realtime_user_window_sec
            )
            physical_devices = {
                physical_device_key(device.get("device_id", "unknown"))
                for device in visible_devices
            }
            realtime_physical_devices = {
                physical_device_key(device.get("device_id", "unknown"))
                for device in visible_devices
                if float(device.get("last_seen_age_sec", realtime_user_window_sec + 1)) <= realtime_user_window_sec
            }

            schedule_order = sorted(
                [
                    item for item in self._recent_requests
                    if item.get("status") in {"queued", "running"}
                ],
                key=lambda item: (
                    0 if item.get("status") == "running" else 1,
                    -float(item.get("dao_priority") or 0),
                    float(item.get("arrived_at") or 0),
                ),
            )[:20]

            planned_gantt = []
            actual_gantt = []
            for item in reversed(list(self._recent_requests)[:40]):
                planned_gantt.extend(item.get("planned_schedule") or [])
                actual_gantt.extend(item.get("actual_schedule") or [])
            lanes = [
                "Network Upload",
                "CPU Decode",
                "GPU YOLO",
                "CPU Track",
                "GPU Pose",
                "GPU State",
                "Network Response",
            ]
            starts = (
                [float(item.get("planned_start", now)) for item in planned_gantt[-180:]]
                + [float(item.get("actual_start", now)) for item in actual_gantt[-180:]]
            )
            ends = (
                [float(item.get("planned_end", now + 1)) for item in planned_gantt[-180:]]
                + [float(item.get("actual_end", now + 1)) for item in actual_gantt[-180:]]
            )

            return {
                "timestamp": now,
                "server": {
                    "node_id": "windows-edge-server",
                    "grpc_port": 50051,
                    "admin_port": 8011,
                    "device": self._device,
                    "torch_device": self._torch_device,
                    "max_workers": self._max_workers,
                },
                "summary": {
                    "total_requests": self._total_requests,
                    "connected_devices": len(physical_devices),
                    "realtime_devices": len(realtime_physical_devices),
                    "realtime_users": realtime_users,
                    "realtime_user_window_sec": realtime_user_window_sec,
                    "active_requests": self._active_requests,
                    "pending_requests": self._pending_requests,
                    "queue_waiting": queue_waiting,
                    "avg_latency_ms": round(avg_latency, 1),
                    "slo_ms": round(self._dao_slo_ms, 1),
                    "slo_met_count": slo_met_count,
                    "slo_window_count": len(completed_recent),
                    "slo_rate": round(slo_rate, 1),
                    "gpu_util": resources["gpu_util"],
                    "idle_workers": resources["idle_workers"],
                    "resource_state": resources["resource_state"],
                },
                "resources": resources,
                "dao": {
                    "policy": "DAGAwareOffloadPolicy",
                    "slo_ms": round(self._dao_slo_ms, 1),
                    "congestion_window_ms": round(self._dao_congestion_window_ms, 1),
                    "max_offloads_per_window": self._dao_max_offloads_per_window,
                    "network_bandwidth_mbps": round(self._dao_bandwidth_mbps, 1),
                    "utilization_threshold": round(self._dao_utilization_threshold, 3),
                    "bottleneck_ratio": round(self._dao_bottleneck_ratio, 3),
                    "current_window_offloads": min(self._pending_requests, self._dao_max_offloads_per_window),
                    "admission_rule": "服务器不忙时按顺序接收任务；繁忙时优先处理更接近超时的请求。",
                },
                "active_requests": self._active_requests,
                "gpu_util": resources["gpu_util"],
                "queue_waiting": queue_waiting,
                "pending_order": list(self._pending_order),
                "schedule_order": schedule_order,
                "workflow": self._workflow_template(),
                "gantt": {
                    "lanes": lanes,
                    "planned": planned_gantt[-180:],
                    "actual": actual_gantt[-180:],
                    "window_start": min([now] + starts),
                    "window_end": max([now + 1] + ends),
                },
                "running_requests": list(self._running_requests.values()),
                "recent_requests": list(self._recent_requests)[:800],
                "devices": visible_devices,
                "stage_offload": dict(self._stage_offload),
                "education": self._latest_education,
            }


def _post_json(url: str, payload: dict) -> None:
    """发送 JSON POST。"""
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=2.0) as resp:
        resp.read()


def start_monitor_reporter(
    service: InferenceServiceImpl,
    monitor_url: str,
    interval_sec: float = 1.0,
) -> tuple[threading.Event, threading.Thread]:
    """定时向集中监控页上报服务器状态。"""
    stop_event = threading.Event()
    endpoint = f"{monitor_url.rstrip('/')}/ingest/server"

    def _run() -> None:
        while not stop_event.is_set():
            try:
                _post_json(endpoint, service.monitor_snapshot())
            except (OSError, error.URLError, TimeoutError) as exc:
                logger.debug(f"monitor 上报失败: {exc}")
            stop_event.wait(interval_sec)

    thread = threading.Thread(target=_run, name="monitor-reporter", daemon=True)
    thread.start()
    return stop_event, thread


def _synthetic_stress_frame(width: int, height: int, index: int, stream_id: int) -> np.ndarray:
    frame = np.full((height, width, 3), (238, 243, 248), dtype=np.uint8)
    phase = (index * 13 + stream_id * 41) % max(width, 1)
    cx = int(width * 0.35 + (phase / max(width, 1) - 0.5) * width * 0.18)
    cy = int(height * 0.48)
    scale = max(0.8, min(width, height) / 480.0)

    cv2.circle(frame, (cx, int(cy - 95 * scale)), int(34 * scale), (70, 102, 150), -1)
    cv2.rectangle(
        frame,
        (int(cx - 42 * scale), int(cy - 55 * scale)),
        (int(cx + 42 * scale), int(cy + 75 * scale)),
        (37, 99, 235),
        -1,
    )
    cv2.line(
        frame,
        (int(cx - 42 * scale), int(cy - 25 * scale)),
        (int(cx - 100 * scale), int(cy + 18 * scale)),
        (70, 102, 150),
        int(10 * scale),
    )
    cv2.line(
        frame,
        (int(cx + 42 * scale), int(cy - 25 * scale)),
        (int(cx + 100 * scale), int(cy + 18 * scale)),
        (70, 102, 150),
        int(10 * scale),
    )
    cv2.line(
        frame,
        (int(cx - 22 * scale), int(cy + 75 * scale)),
        (int(cx - 42 * scale), int(cy + 155 * scale)),
        (70, 102, 150),
        int(12 * scale),
    )
    cv2.line(
        frame,
        (int(cx + 22 * scale), int(cy + 75 * scale)),
        (int(cx + 42 * scale), int(cy + 155 * scale)),
        (70, 102, 150),
        int(12 * scale),
    )
    cv2.putText(
        frame,
        f"GPU STRESS stream={stream_id + 1} frame={index}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    return frame


def _encode_jpeg(frame: np.ndarray, quality: int = 86) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode stress frame")
    return encoded.tobytes()


def _load_stress_frames(
    video_path: str | None,
    frame_count: int,
    width: int,
    height: int,
) -> list[bytes]:
    frames: list[bytes] = []
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            while len(frames) < frame_count:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height))
                frames.append(_encode_jpeg(frame))
            cap.release()
        if frames:
            logger.info(f"GPU stress loaded {len(frames)} frames from {video_path}")
            return frames
        logger.warning(f"GPU stress video unavailable or empty: {video_path}; using synthetic frames")

    for i in range(frame_count):
        frames.append(_encode_jpeg(_synthetic_stress_frame(width, height, i, 0)))
    logger.info(f"GPU stress generated {len(frames)} synthetic frames")
    return frames


def start_gpu_stress(
    service: InferenceServiceImpl,
    streams: int,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    video_path: str | None,
    duration_sec: float,
) -> tuple[threading.Event, list[threading.Thread]]:
    """Generate local offload requests to keep the server GPU busy for diagnosis."""
    stop_event = threading.Event()
    frames = _load_stress_frames(video_path, max(1, frame_count), width, height)
    interval = 0.0 if fps <= 0 else 1.0 / fps
    stream_count = max(1, int(streams))

    def _run_stream(stream_id: int) -> None:
        frame_index = 0
        started = time.time()
        while not stop_event.is_set():
            if duration_sec > 0 and time.time() - started >= duration_sec:
                break
            jpeg = frames[(frame_index + stream_id) % len(frames)]
            request_msg = pb2.PipelineRequest(
                jpeg_frame=jpeg,
                frame_index=frame_index,
                dag_id="gpu-stress-dag",
                deadline_ms=float(service._dao_slo_ms),
                device_id=f"stress-device-{stream_id + 1:02d}",
            )
            try:
                service.RunPipelineA(request_msg, None)
            except Exception as exc:
                logger.warning(f"GPU stress stream {stream_id + 1} failed: {exc}")
                time.sleep(0.2)
            frame_index += 1
            if interval > 0:
                stop_event.wait(interval)

    threads = [
        threading.Thread(
            target=_run_stream,
            name=f"gpu-stress-{index + 1}",
            args=(index,),
            daemon=True,
        )
        for index in range(stream_count)
    ]
    for thread in threads:
        thread.start()
    logger.info(
        "GPU stress enabled: "
        f"streams={stream_count}, fps={fps}, frames={len(frames)}, "
        f"size={width}x{height}, duration_sec={duration_sec or 'until-stop'}"
    )
    return stop_event, threads


SERVER_ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>边缘服务器 DAO 调度管理端</title>
  <style>
    :root {
      color: #102033;
      background: #eef4f8;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 1220px; }
    .shell { width: min(1540px, calc(100vw - 44px)); margin: 0 auto; padding: 26px 0 36px; }
    header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px; }
    .eyebrow { margin: 0 0 6px; color: #2563eb; font-weight: 900; letter-spacing: .06em; }
    h1 { margin: 0; font-size: 34px; letter-spacing: 0; }
    header span { display: block; margin-top: 8px; color: #667085; }
    .pill { padding: 10px 13px; border-radius: 999px; background: #d1fae5; color: #047857; font-weight: 900; }
    .stats, .resource-grid, .dao-grid, .workflow, .main-grid, .two-grid { display: grid; gap: 14px; }
    .stats { grid-template-columns: repeat(7, 1fr); margin-bottom: 14px; }
    .resource-grid { grid-template-columns: repeat(4, 1fr); }
    .dao-grid { grid-template-columns: repeat(6, 1fr); }
    .workflow { grid-template-columns: repeat(7, 1fr); }
    .main-grid { grid-template-columns: 360px 1fr 420px; align-items: start; }
    .two-grid { grid-template-columns: 1.2fr .8fr; align-items: start; }
    .card, .panel {
      background: rgba(255, 255, 255, .94);
      border: 1px solid #dbe4ee;
      border-radius: 8px;
      box-shadow: 0 18px 38px rgba(15, 23, 42, .08);
    }
    .card { min-height: 116px; padding: 18px; }
    .panel { padding: 16px; margin-bottom: 14px; }
    small, .muted { color: #667085; }
    .card small, .metric small { display: block; font-weight: 800; }
    .card strong { display: block; margin-top: 15px; font-size: 30px; }
    .metric {
      padding: 13px;
      border: 1px solid #dbe4ee;
      border-radius: 8px;
      background: #f8fafc;
      min-height: 96px;
    }
    .metric strong { display: block; margin-top: 8px; font-size: 21px; }
    .heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
    .heading b { font-size: 18px; }
    .heading span { padding: 5px 9px; border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-size: 12px; font-weight: 900; }
    .bar { height: 8px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin-top: 8px; }
    .bar i { display: block; height: 100%; background: linear-gradient(90deg, #2563eb, #10b981); }
    .bar.warn i { background: linear-gradient(90deg, #f59e0b, #ef4444); }
    .flow-step, .device, .task, .rank-item {
      padding: 12px;
      border: 1px solid #dbe4ee;
      border-radius: 8px;
      background: #f8fafc;
    }
    .flow-step { position: relative; min-height: 116px; display: flex; flex-direction: column; }
    .flow-step p.muted { flex: 1 1 auto; margin: 0; padding-bottom: 38px; }
    .flow-step .tag { position: absolute; left: 12px; bottom: 12px; }
    .flow-step:not(:last-child)::after {
      content: "";
      position: absolute;
      right: -14px;
      top: 50%;
      width: 14px;
      height: 2px;
      background: #94a3b8;
    }
    .flow-step strong { display: block; margin: 8px 0; }
    .list { display: grid; gap: 10px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .tag { padding: 5px 8px; border-radius: 999px; background: #e2e8f0; color: #334155; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .tag.running { background: #d1fae5; color: #047857; }
    .tag.queued { background: #dbeafe; color: #1d4ed8; }
    .tag.completed { background: #e0e7ff; color: #3730a3; }
    .tag.failed { background: #fee2e2; color: #b91c1c; }
    .task-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; }
    .task-grid span { padding: 9px; border-radius: 7px; background: #fff; border: 1px solid #e2e8f0; font-size: 13px; }
    .empty { padding: 22px; color: #667085; text-align: center; border: 1px dashed #cbd5e1; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }
    th, td { padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 13px; }
    th { color: #667085; background: #f8fafc; }
    .gantt-board { display: grid; gap: 10px; }
    .gantt-row {
      display: grid;
      grid-template-columns: 180px repeat(5, minmax(120px, 1fr)) 150px;
      gap: 8px;
      align-items: stretch;
    }
    .gantt-head {
      color: #667085;
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }
    .gantt-task {
      padding: 10px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #dbe4ee;
      min-height: 76px;
    }
    .gantt-task strong { display: block; margin-bottom: 6px; }
    .gantt-cell {
      padding: 9px;
      border-radius: 8px;
      background: #fff;
      border: 1px solid #dbe4ee;
      min-height: 76px;
    }
    .gantt-cell b { display: block; font-size: 13px; margin-bottom: 8px; color: #334155; }
    .dual-bars { display: grid; gap: 5px; }
    .mini-line { display: grid; grid-template-columns: 34px 1fr 62px; align-items: center; gap: 6px; font-size: 11px; color: #667085; }
    .mini-track { height: 8px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
    .mini-track i { display: block; height: 100%; border-radius: 999px; }
    .mini-track.plan i { background: #2563eb; }
    .mini-track.real i { background: #059669; }
    .gantt-summary {
      padding: 10px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #dbe4ee;
      min-height: 76px;
      font-size: 13px;
    }
    .gantt-summary strong { display: block; font-size: 18px; margin: 4px 0; }
    .legend { display: flex; gap: 14px; color: #667085; font-size: 13px; }
    .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 5px; }
    .blue { background: #2563eb; }
    .green { background: #059669; }
    .footer-note { margin-top: 10px; color: #667085; font-size: 13px; }
    /* ── 一级标签页导航（参考 Challenge-cup 管理端风格） ── */
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px; }
    .tab {
      padding: 10px 20px;
      border: 1px solid #dbe4ee;
      border-radius: 999px;
      background: rgba(255, 255, 255, .94);
      color: #475569;
      font-size: 14px;
      font-weight: 800;
      font-family: inherit;
      cursor: pointer;
      transition: border-color .15s ease, color .15s ease, background .15s ease;
    }
    .tab:hover { border-color: #2563eb; color: #1d4ed8; }
    .tab.active {
      background: #2563eb;
      border-color: #2563eb;
      color: #fff;
      box-shadow: 0 8px 18px rgba(37, 99, 235, .28);
    }
    .view { display: none; }
    .view.active { display: block; }
    /* ── 甘特图子栏：三图自由勾选 ── */
    .chart-toggles {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      margin-bottom: 14px; padding: 10px 14px;
      border: 1px solid #dbe4ee; border-radius: 10px;
      background: rgba(255, 255, 255, .8);
    }
    .chart-toggles .toggles-label { font-size: 12px; font-weight: 900; color: #667085; letter-spacing: .05em; }
    .toggle-chip {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 7px 14px;
      border: 1px solid #dbe4ee;
      border-radius: 999px;
      background: #fff;
      font-size: 13px; font-weight: 800; color: #475569;
      cursor: pointer; user-select: none;
      transition: border-color .15s ease, color .15s ease;
    }
    .toggle-chip:hover { border-color: #2563eb; color: #1d4ed8; }
    .toggle-chip input { accent-color: #2563eb; width: 15px; height: 15px; margin: 0; cursor: pointer; }
    /* ── New Gantt: Device Heat Map & Stream Timeline ── */
    .new-gantt-grid { display: grid; grid-template-columns: 1fr; gap: 14px; margin-bottom: 14px; }
    .heat-wrapper { display: flex; gap: 0; overflow: hidden; border: 1px solid #dbe4ee; border-radius: 8px; margin-top: 12px; }
    .heat-y-axis { flex: 0 0 120px; display: flex; flex-direction: column; background: #f8fafc; border-right: 1px solid #dbe4ee; }
    .heat-axis-spacer { height: 24px; border-bottom: 1px solid #dbe4ee; }
    .heat-y-label { height: 36px; display: flex; align-items: center; padding: 0 10px; border-bottom: 1px solid #edf2f7; font-size: 11px; font-weight: 800; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .heat-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
    .heat-x-axis { position: relative; height: 24px; background: #f8fafc; border-bottom: 1px solid #dbe4ee; }
    .heat-x-axis span { position: absolute; transform: translateX(-50%); top: 5px; font-size: 10px; color: #667085; font-weight: 700; }
    .heat-rows { display: flex; flex-direction: column; position: relative; overflow: hidden; }
    .heat-rows::after, .stream-area::after, .task-gantt-track::after {
      content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 2;
      background: linear-gradient(90deg, #ffffff 0, rgba(255, 255, 255, 0) 2.5%, rgba(255, 255, 255, 0) 97.5%, #ffffff 100%);
    }
    .heat-dev-row { position: relative; height: 36px; border-bottom: 1px solid #edf2f7; }
    .heat-cell {
      position: absolute;
      top: 3px;
      height: 30px;
      width: calc(3.333% - 2px);
      border-radius: 3px;
    }
    .heat-cell.hc-idle { background: #f1f5f9; }
    .heat-cell.hc-running { background: #2563eb; box-shadow: 0 2px 5px rgba(37, 99, 235, .35); }
    .heat-cell.hc-completed { background: #059669; box-shadow: 0 2px 5px rgba(5, 150, 105, .35); }
    .heat-cell.hc-queued { background: #f59e0b; box-shadow: 0 2px 5px rgba(245, 158, 11, .35); }
    .heat-slide { position: relative; animation: task-slide var(--slide-dur, 1s) linear forwards; }
    .stream-area { display: flex; flex-direction: column; position: relative; overflow: hidden; }
    .stream-dev-row { position: relative; height: 36px; border-bottom: 1px solid #edf2f7; background: repeating-linear-gradient(90deg, transparent, transparent calc(10% - 1px), #f1f5f9 calc(10% - 1px), #f1f5f9 10%); }
    .stream-frame-bar { position: absolute; top: 8px; height: 20px; min-width: 3px; border-radius: 4px; }
    .stream-frame-bar.sb-running { background: linear-gradient(135deg, #1d4ed8, #3b82f6); box-shadow: 0 2px 6px rgba(29,78,216,0.35); }
    .stream-frame-bar.sb-completed { background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 2px 6px rgba(5,150,105,0.35); }
    .stream-frame-bar.sb-queued { background: linear-gradient(135deg, #d97706, #f59e0b); box-shadow: 0 2px 6px rgba(217,119,6,0.35); }
    .stream-frame-bar.sb-failed { background: linear-gradient(135deg, #dc2626, #ef4444); box-shadow: 0 2px 6px rgba(220,38,38,0.35); }
    .new-gantt-legend { display: flex; align-items: center; gap: 14px; margin-top: 10px; font-size: 12px; color: #667085; }
    .new-gantt-legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 5px; }
    .task-gantt { display: grid; gap: 10px; margin-top: 14px; }
    .task-gantt-scale { position: relative; height: 26px; margin-left: 190px; border-bottom: 1px solid #dbe4ee; }
    .task-gantt-scale span { position: absolute; top: 5px; transform: translateX(-50%); color: #667085; font-size: 11px; font-weight: 800; }
    .task-gantt-row { display: grid; grid-template-columns: 180px 1fr; gap: 10px; align-items: stretch; }
    .task-gantt-device { padding: 10px; border: 1px solid #dbe4ee; border-radius: 8px; background: #f8fafc; min-height: 122px; }
    .task-gantt-device strong { display: block; font-size: 13px; margin-bottom: 5px; }
    .task-gantt-device span { color: #667085; font-size: 11px; word-break: break-all; }
    .task-gantt-track {
      position: relative;
      min-height: 122px;
      border: 1px solid #dbe4ee;
      border-radius: 8px;
      background: repeating-linear-gradient(90deg, #ffffff, #ffffff calc(10% - 1px), #f1f5f9 calc(10% - 1px), #f1f5f9 10%);
      overflow: hidden;
    }
    .task-gantt-bars {
      position: absolute; inset: 0;
      animation: task-slide var(--slide-dur, 1s) linear forwards;
    }
    @keyframes task-slide {
      from { transform: translateX(0); }
      to { transform: translateX(calc(-1 * var(--slide-pct, 0%))); }
    }
    .task-stage-bar {
      position: absolute;
      height: 20px;
      top: 8px;
      min-width: 10px;
      border-radius: 5px;
      color: #fff;
      font-size: 12px;
      font-weight: 900;
      line-height: 20px;
      padding: 0;
      white-space: nowrap;
      overflow: hidden;
      box-shadow: 0 6px 14px rgba(15, 23, 42, .16);
    }
    .task-stage-bar:nth-child(4n+2) { top: 34px; }
    .task-stage-bar:nth-child(4n+3) { top: 60px; }
    .task-stage-bar:nth-child(4n) { top: 86px; }
    .task-stage-plan { opacity: .42; border: 1px dashed rgba(15, 23, 42, .32); box-shadow: none; }
    .stage-upload { background: #64748b; }
    .stage-decode { background: #0f766e; }
    .stage-yolo { background: #2563eb; }
    .stage-track { background: #7c3aed; }
    .stage-pose { background: #db2777; }
    .stage-state { background: #ea580c; }
    .stage-response { background: #059669; }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <p class="eyebrow">边缘服务器 · 管理台</p>
        <h1>边缘服务器智能调度管理台</h1>
        <span>实时查看用户接入、服务器负载、画面处理流程与排队顺序。</span>
      </div>
      <div class="pill" id="syncTime">等待同步</div>
    </header>

    <section class="stats" id="stats"></section>

    <section class="panel">
      <div class="heading"><span>处理流程</span><b>一帧画面要经过的处理步骤</b></div>
      <div class="workflow" id="workflow"></div>
    </section>

    <nav class="tabs" aria-label="管理视图">
      <button class="tab active" type="button" data-view="view-resources">资源</button>
      <button class="tab" type="button" data-view="view-dao">排队规则</button>
      <button class="tab" type="button" data-view="view-gantt">时间线</button>
      <button class="tab" type="button" data-view="view-devices">用户 · 队列 · 优先级</button>
      <button class="tab" type="button" data-view="view-history">历史记录</button>
    </nav>

    <section id="view-resources" class="view active">
      <section class="panel">
        <div class="heading"><span>资源</span><b>服务器硬件实时占用</b></div>
        <div class="resource-grid" id="resources"></div>
      </section>
    </section>

    <section id="view-dao" class="view">
      <section class="panel">
        <div class="heading"><span>排队规则</span><b>智能排队与优先处理参数</b></div>
        <div class="dao-grid" id="daoMetrics"></div>
        <p class="footer-note" id="daoRule"></p>
      </section>
    </section>

    <section id="view-gantt" class="view">
      <div class="chart-toggles">
        <span class="toggles-label">选择要显示的图表</span>
        <label class="toggle-chip"><input type="checkbox" data-chart="panel-gantt" checked>耗时对比（计划 vs 实际）</label>
        <label class="toggle-chip"><input type="checkbox" data-chart="panel-task-gantt" checked>分步处理时间线</label>
        <label class="toggle-chip"><input type="checkbox" data-chart="panel-device-timeline" checked>用户活跃与帧处理</label>
      </div>
      <section class="panel" id="panel-gantt">
        <div class="heading">
          <span>耗时对比</span>
          <b>每个请求的计划耗时与实际耗时</b>
        </div>
        <div class="legend"><span><i class="blue"></i>计划耗时</span><span><i class="green"></i>实际耗时</span></div>
        <div class="gantt-board" id="gantt"></div>
        <p class="footer-note">每行代表一位用户上传的一帧画面，每列代表一个处理步骤。蓝色为计划耗时，绿色为实际耗时，可以对比多人同时使用时的排队与快慢。</p>
      </section>

      <section class="panel" id="panel-task-gantt">
        <div class="heading">
          <span>分步时间线</span>
          <b>每位用户画面帧的处理时段</b>
        </div>
        <div class="new-gantt-legend">
          <span><i style="background:#2563eb"></i>人物检测</span>
          <span><i style="background:#db2777"></i>姿态分析</span>
          <span><i style="background:#ea580c"></i>状态识别</span>
          <span><i style="background:#059669"></i>结果返回</span>
          <span><i style="background:#94a3b8"></i>半透明为计划时段</span>
        </div>
        <div id="taskStageGantt"></div>
        <p class="footer-note">每位用户一行，显示最近画面帧在各处理步骤上的占用时段，例如“用户1 人物检测”“用户2 姿态分析”。多人同时使用时，可以看到服务器如何交错安排任务。</p>
      </section>

      <div class="new-gantt-grid" id="panel-device-timeline">
        <section class="panel">
          <div class="heading"><span>用户活跃</span><b>用户活跃热力图</b></div>
          <div class="new-gantt-legend"><span><i style="background:#2563eb"></i>执行中</span><span><i style="background:#059669"></i>已完成</span><span><i style="background:#f59e0b"></i>排队中</span><span><i style="background:#f8fafc;border:1px solid #dbe4ee"></i>空闲</span></div>
          <div id="deviceHeatMap"></div>
          <p class="footer-note">每格代表 2 秒，显示最近 1 分钟内每位用户的画面发送密度与状态，颜色越深表示任务越多。</p>
        </section>
        <section class="panel">
          <div class="heading"><span>帧处理</span><b>每帧画面到达与处理快慢</b></div>
          <div class="new-gantt-legend"><span><i style="background:#1d4ed8"></i>执行中</span><span><i style="background:#059669"></i>已完成</span><span><i style="background:#d97706"></i>排队</span><span><i style="background:#dc2626"></i>失败</span></div>
          <div id="streamTimeline"></div>
          <p class="footer-note">每条横杠代表一帧画面请求：位置是到达时间，宽度对应处理耗时，越长表示处理越慢。</p>
        </section>
      </div>
    </section>

    <section id="view-devices" class="view">
      <main class="main-grid">
        <section class="panel">
          <div class="heading"><span>用户</span><b>用户列表</b></div>
          <div class="list" id="devices"></div>
        </section>
        <section class="panel">
          <div class="heading"><span>队列</span><b>正在排队和处理的任务</b></div>
          <div class="list" id="activeTasks"></div>
        </section>
        <section class="panel">
          <div class="heading"><span>优先级</span><b>接下来先处理谁</b></div>
          <div class="list" id="rank"></div>
        </section>
      </main>
    </section>

    <section id="view-history" class="view">
      <section class="panel">
        <div class="heading"><span>历史记录</span><b>最近处理记录</b></div>
        <div id="history"></div>
      </section>
    </section>
  </div>

  <script>
    const byId = id => document.getElementById(id);
    const fmt = value => {
      const n = Number(value || 0);
      return Number.isFinite(n) ? String(Math.round(n * 10) / 10) : "0";
    };
    const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[ch]));
    const statusText = status => ({
      queued: "排队中",
      running: "执行中",
      completed: "已完成",
      failed: "失败",
      defer: "延后"
    })[status] || "等待";
    const friendlyDevice = id => {
      const raw = String(id || "");
      if (!raw || raw === "-" || raw === "unknown") return "未知用户";
      const name = raw.replace(/^client_/i, "").split("__")[0];
      return name ? `用户${name}` : "未知用户";
    };
    const stateLabel = state => ({
      focused: "专注", distracted: "分心", tired: "疲惫", absent: "离开", unknown: "待识别"
    })[state] || state || "-";
    const stageLabel = stage => ({
      "Frame Upload": "上传画面帧", "Decode": "解码画面", "YOLO Detect": "人物目标检测",
      "Track": "多目标跟踪", "ReID": "行人重识别", "YOLO-pose": "人体姿态估计",
      "State Fusion": "学习状态融合", "Response": "返回识别结果",
      "YOLO": "人物目标检测", "Pose": "人体姿态估计", "状态融合": "学习状态融合",
      "YOLO + Pose + State Fusion": "人物检测 + 姿态估计 + 状态融合", "waiting": "排队等待处理"
    })[stage] || stage || "-";
    const admissionLabel = value => ({
      accept: "已接收", defer: "已延后"
    })[value] || statusText(value);
    const metric = (label, value, sub = "") => `
      <article class="card"><small>${esc(label)}</small><strong>${value}</strong><span>${esc(sub)}</span></article>`;
    const smallMetric = (label, value, sub = "", warn = false) => `
      <article class="metric">
        <small>${esc(label)}</small><strong>${value}</strong><span class="muted">${esc(sub)}</span>
        <div class="bar ${warn ? "warn" : ""}"><i style="width:${Math.max(0, Math.min(100, Number(value) || 0))}%"></i></div>
      </article>`;

    function renderCards(snapshot) {
      const s = snapshot.summary || {};
      byId("stats").innerHTML = [
        metric("在线用户", s.connected_devices || 0, `正在使用 ${s.realtime_devices || 0} 人`),
        metric("活跃用户", s.realtime_users || 0, `最近 ${fmt(s.realtime_user_window_sec || 15)} 秒上传视频的用户`),
        metric("累计处理请求", s.total_requests || 0, "服务器收到的画面帧总数"),
        metric("执行中", s.active_requests || 0, "正在分析的画面数"),
        metric("排队等待", s.queue_waiting || 0, "按优先级排队等待"),
        metric("平均延迟", `${fmt(s.avg_latency_ms)} ms`, "最近完成一帧的平均耗时"),
        metric("按时完成率", `${fmt(s.slo_rate)}%`, `最近 ${s.slo_window_count || 0} 帧中 ${s.slo_met_count || 0} 帧按时完成`),
      ].join("");
    }

    function renderResources(snapshot) {
      const r = snapshot.resources || {};
      byId("resources").innerHTML = [
        smallMetric("CPU 利用率", fmt(r.cpu_util), `${r.resource_state || "-"} · 最忙环节占用 ${fmt((r.bottleneck || 0) * 100)}%`, Number(r.cpu_util) > 80),
        smallMetric("GPU 利用率", fmt(r.gpu_util), r.gpu_name || "GPU 估计值", Number(r.gpu_util) > 80),
        smallMetric("GPU 显存", fmt(r.gpu_memory_util), `${fmt(r.gpu_memory_used_mb)} / ${fmt(r.gpu_memory_total_mb)} MB`, Number(r.gpu_memory_util) > 80),
        smallMetric("内存利用率", fmt(r.memory_util), `${fmt(r.memory_used_gb)} / ${fmt(r.memory_total_gb)} GB`, Number(r.memory_util) > 80),
        metric("空闲处理通道", `${r.idle_workers || 0} / ${r.max_workers || 0}`, `正在处理 ${r.active_workers || 0} 个，排队 ${r.queue_waiting || 0} 个`),
        metric("资源可用性", r.available ? "可用" : "高负载", r.resource_state || "-"),
      ].join("");
    }

    function renderDao(snapshot) {
      const d = snapshot.dao || {};
      byId("daoMetrics").innerHTML = [
        metric("延迟目标", `${fmt(d.slo_ms)} ms`, "每帧画面应在此时间内完成"),
        metric("并发控制窗口", `${fmt(d.congestion_window_ms)} ms`, "繁忙时按此周期控制放量"),
        metric("每周期最多放行", d.max_offloads_per_window || 0, "一个周期内允许开始处理的请求数"),
        metric("网络带宽", `${fmt(d.network_bandwidth_mbps)} Mbps`, "按网速估算画面上传耗时"),
        metric("负载警戒线", `${fmt((d.utilization_threshold || 0) * 100)}%`, "负载超过后减少接收新任务"),
        metric("卡顿判断线", `${fmt((d.bottleneck_ratio || 0) * 100)}%`, "用于找出拖慢整体的处理环节"),
      ].join("");
      byId("daoRule").textContent = d.admission_rule || "系统根据任务优先级、延迟要求、排队情况和服务器负载，自动决定先处理谁、是否稍后处理。";
    }

    function renderWorkflow(snapshot) {
      const workflow = snapshot.workflow || [];
      const stepNames = {
        "Frame Upload": "上传画面", "Decode": "解码图像", "YOLO Detect": "人物检测",
        "Track": "目标跟踪", "YOLO-pose": "姿态分析", "State Fusion": "状态识别", "Response": "返回结果",
      };
      const stepResources = {
        "Network Upload": "网络上传", "CPU Decode": "CPU 解码", "GPU YOLO": "GPU 检测",
        "CPU Track": "CPU 跟踪", "GPU Pose": "GPU 姿态", "GPU State": "GPU 识别", "Network Response": "网络返回",
      };
      byId("workflow").innerHTML = workflow.map(step => `
        <article class="flow-step">
          <small>${esc(stepResources[step.resource] || step.resource)}</small>
          <strong>${esc(stepNames[step.name] || step.name)}</strong>
          <p class="muted">${esc(step.desc)}</p>
          <span class="tag">${fmt(step.duration_ms)} ms</span>
        </article>`).join("") || `<div class="empty">暂无处理流程</div>`;
    }

    function renderGantt(snapshot) {
      const recent = (snapshot.recent_requests || [])
        .filter(task => task.planned_schedule || task.actual_schedule || task.stage_timings)
        .slice(0, 20);
      const stages = [
        { key: "YOLO Detect", label: "人物检测" },
        { key: "Track", label: "目标跟踪" },
        { key: "YOLO-pose", label: "姿态分析" },
        { key: "State Fusion", label: "状态识别" },
        { key: "Response", label: "返回结果" },
      ];
      if (!recent.length) {
        byId("gantt").innerHTML = `<div class="empty">等待用户上传画面，生成处理计划</div>`;
        return;
      }
      const plannedValues = [];
      const actualValues = [];
      for (const task of recent) {
        for (const item of task.planned_schedule || []) plannedValues.push(Number(item.duration_ms || 0));
        for (const item of task.actual_schedule || []) actualValues.push(Number(item.duration_ms || 0));
        for (const value of Object.values(task.stage_timings || {})) actualValues.push(Number(value || 0));
      }
      const maxStageMs = Math.max(20, ...plannedValues, ...actualValues);
      const findPlanned = (task, stageKey) => (task.planned_schedule || []).find(item => item.stage === stageKey);
      const findActual = (task, stageKey) => {
        const direct = (task.actual_schedule || []).find(item => item.stage === stageKey);
        if (direct) return direct;
        const fallbackMap = {
          "YOLO Detect": "YOLO",
          "Track": "Track",
          "YOLO-pose": "Pose",
          "State Fusion": "状态融合",
        };
        const value = (task.stage_timings || {})[fallbackMap[stageKey]];
        return value ? { duration_ms: value } : null;
      };
      const cellHtml = (task, stage) => {
        const planned = findPlanned(task, stage.key);
        const actual = findActual(task, stage.key);
        const planMs = Number(planned?.duration_ms || 0);
        const realMs = Number(actual?.duration_ms || 0);
        const cellMax = Math.max(planMs, realMs, 1);
        const planWidth = planMs > 0 ? Math.max(8, Math.min(100, planMs / cellMax * 100)) : 0;
        const realWidth = realMs > 0 ? Math.max(8, Math.min(100, realMs / cellMax * 100)) : 0;
        return `
          <div class="gantt-cell">
            <b>${esc(stage.label)}</b>
            <div class="dual-bars">
              <div class="mini-line"><span>计划</span><div class="mini-track plan"><i style="width:${planWidth}%"></i></div><span>${fmt(planMs)}ms</span></div>
              <div class="mini-line"><span>实际</span><div class="mini-track real"><i style="width:${realWidth}%"></i></div><span>${realMs ? fmt(realMs) + "ms" : "-"}</span></div>
            </div>
          </div>`;
      };
      const rows = recent.map(task => {
        const planTotal = Number(task.planned_total_ms || 0);
        const realTotal = Number(task.total_ms || 0);
        const sloText = task.slo_met === false ? `超时 ${fmt(task.slo_over_ms)}ms` : task.slo_met === true ? "按时完成" : "执行中";
        return `
          <div class="gantt-row">
            <div class="gantt-task">
              <strong>帧 #${task.sequence || "-"} · ${esc(friendlyDevice(task.device_id))}</strong>
              <div class="muted">帧号 ${esc(task.frame_index ?? "-")} · ${statusText(task.status)}</div>
              <div class="muted">优先级 ${fmt((task.dao_priority || 0) * 100)}%</div>
            </div>
            ${stages.map(stage => cellHtml(task, stage)).join("")}
            <div class="gantt-summary">
              <span class="muted">实际总耗时</span>
              <strong>${fmt(realTotal)} ms</strong>
              <div class="muted">计划 ${fmt(planTotal)} ms</div>
              <div class="muted">${esc(sloText)}</div>
            </div>
          </div>`;
      }).join("");
      const header = `
        <div class="gantt-row gantt-head">
          <div>请求</div>
          ${stages.map(stage => `<div>${esc(stage.label)}</div>`).join("")}
          <div>结果</div>
        </div>`;
      byId("gantt").innerHTML = header + rows;
    }

    function renderDevices(snapshot) {
      const devices = snapshot.devices || [];
      byId("devices").innerHTML = devices.map(device => `
        <article class="device">
          <div class="row"><strong>${esc(friendlyDevice(device.device_id))}</strong><span class="tag ${esc(device.last_status)}">${statusText(device.last_status)}</span></div>
          <p class="muted">最近一帧 ${esc(device.last_frame ?? "-")} · ${fmt(device.last_seen_age_sec)} 秒前</p>
          <div class="task-grid">
            <span>累计 ${device.total_requests || 0}</span>
            <span>排队 ${device.queued || 0}</span>
            <span>运行 ${device.running || 0}</span>
            <span>完成 ${device.completed || 0}</span>
          </div>
        </article>`).join("") || `<div class="empty">等待用户设备连接</div>`;
    }

    function taskHtml(task) {
      return `
        <article class="task">
          <div class="row"><strong>帧 #${task.sequence || "-"} · ${esc(friendlyDevice(task.device_id))}</strong><span class="tag ${esc(task.status)}">${statusText(task.status)}</span></div>
          <p class="muted">帧号 ${esc(task.frame_index)} · ${esc(stageLabel(task.stage))}</p>
          <div class="task-grid">
            <span>排队 ${fmt(task.queue_wait_ms)} ms</span>
            <span>总耗时 ${fmt(task.total_ms)} ms</span>
            <span>时限 ${fmt(task.slo_ms)} ms</span>
            <span>优先级 ${fmt((task.dao_priority || 0) * 100)}%</span>
          </div>
        </article>`;
    }

    function renderQueues(snapshot) {
      const recent = snapshot.recent_requests || [];
      const active = recent.filter(item => ["queued", "running"].includes(item.status));
      byId("activeTasks").innerHTML = active.map(taskHtml).join("") || `<div class="empty">当前没有排队或正在处理的任务</div>`;
      byId("rank").innerHTML = (snapshot.schedule_order || []).map(taskHtml).join("") || `<div class="empty">暂无排队任务，先来先处理</div>`;
    }

    function renderHistory(snapshot) {
      const rows = (snapshot.recent_requests || []).slice(0, 80).map(task => `
        <tr>
          <td>第 ${task.sequence || "-"} 帧</td>
          <td>${esc(friendlyDevice(task.device_id))}</td>
          <td>${esc(task.frame_index)}</td>
          <td><span class="tag ${esc(task.status)}">${statusText(task.status)}</span></td>
          <td>${fmt(task.queue_wait_ms)} ms</td>
          <td>${fmt(task.total_ms)} ms</td>
          <td>${task.slo_met === false ? `超时 ${fmt(task.slo_over_ms)} ms` : task.slo_met === true ? "按时" : "-"}</td>
          <td>${fmt((task.dao_priority || 0) * 100)}%</td>
          <td>${esc(stateLabel(task.learning_state))}</td>
          <td>${esc(admissionLabel(task.admission))}</td>
        </tr>`).join("");
      byId("history").innerHTML = `
        <table>
          <thead><tr><th>序号</th><th>用户</th><th>帧号</th><th>状态</th><th>排队</th><th>总耗时</th><th>时限</th><th>优先级</th><th>学生状态</th><th>接收</th></tr></thead>
          <tbody>${rows || `<tr><td colspan="10" class="muted">暂无处理记录</td></tr>`}</tbody>
        </table>`;
    }

    var lastHeatRenderMs = 0;
    function renderDeviceHeatMap(snapshot) {
      var nowMs = Date.now();
      var target = byId("deviceHeatMap");
      if (nowMs - lastHeatRenderMs < 1900 && target.childElementCount > 0) return;
      lastHeatRenderMs = nowMs;
      var requests = snapshot.recent_requests || [];
      var devices = snapshot.devices || [];
      var now = nowMs / 1000;
      var windowSec = 60;
      var bucketSec = 2;
      var cols = 30;
      var bucketNow = Math.floor(now / bucketSec) * bucketSec;
      var start = bucketNow - (cols - 1) * bucketSec;
      var slidePct = bucketSec / windowSec * 100;
      var devList = devices.length ? devices : [{ device_id: "等待用户" }];
      var buckets = {};
      for (var di = 0; di < devList.length; di++) {
        buckets[devList[di].device_id] = [];
        for (var ci = 0; ci < cols; ci++) buckets[devList[di].device_id].push({ running: 0, completed: 0, queued: 0, failed: 0 });
      }
      for (var ri = 0; ri < requests.length; ri++) {
        var r = requests[ri];
        var ts = Number(r.arrived_at || 0);
        var idx = Math.floor((ts - start) / bucketSec);
        if (idx < 0 || idx >= cols) continue;
        var did = r.device_id || "unknown";
        if (!buckets[did]) continue;
        var st = r.status || "completed";
        if (buckets[did][idx][st] !== undefined) buckets[did][idx][st]++;
        else buckets[did][idx].completed++;
      }
      var timeLabels = [{ pos: "0%", text: "1分钟前" }, { pos: "50%", text: "30秒前" }, { pos: "100%", text: "现在" }];
      var xHtml = '<div class="heat-x-axis">';
      for (var ti = 0; ti < timeLabels.length; ti++) {
        var align = timeLabels[ti].pos === "100%" ? "translateX(-100%)" : (timeLabels[ti].pos === "0%" ? "translateX(0)" : "translateX(-50%)");
        xHtml += '<span style="left:' + timeLabels[ti].pos + ';transform:' + align + '">' + timeLabels[ti].text + '</span>';
      }
      xHtml += '</div>';
      var yHtml = '';
      for (var yi = 0; yi < devList.length; yi++) {
        yHtml += '<div class="heat-y-label">' + esc(friendlyDevice(devList[yi].device_id)) + '</div>';
      }
      var rowsHtml = '';
      for (var dri = 0; dri < devList.length; dri++) {
        var cells = '';
        for (var ci2 = 0; ci2 < cols; ci2++) {
          var c = buckets[devList[dri].device_id][ci2];
          var total = c.running + c.completed + c.queued + c.failed;
          var leftPct = (start + ci2 * bucketSec - (now - windowSec)) / windowSec * 100;
          if (total === 0) {
            cells += '<div class="heat-cell hc-idle" style="left:' + leftPct.toFixed(2) + '%" title="空闲"></div>';
            continue;
          }
          var dominant = (c.running >= c.completed && c.running >= c.queued) ? "running" : (c.completed >= c.queued ? "completed" : "queued");
          var count = c[dominant];
          var opacity = Math.max(0.25, Math.min(1, count / 3));
          cells += '<div class="heat-cell hc-' + dominant + '" style="left:' + leftPct.toFixed(2) + '%;opacity:' + opacity.toFixed(2) + '" title="这 2 秒内 ' + total + ' 个任务（执行 ' + c.running + ' · 完成 ' + c.completed + ' · 排队 ' + c.queued + '）"></div>';
        }
        rowsHtml += '<div class="heat-dev-row"><div class="heat-slide" style="--slide-pct:' + slidePct.toFixed(3) + '%;--slide-dur:2s">' + cells + '</div></div>';
      }
      target.innerHTML = '<div class="heat-wrapper"><div class="heat-y-axis"><div class="heat-axis-spacer"></div>' + yHtml + '</div><div class="heat-main">' + xHtml + '<div class="heat-rows">' + rowsHtml + '</div></div></div>';
    }

    function renderStreamTimeline(snapshot) {
      var requests = snapshot.recent_requests || [];
      var now = Date.now() / 1000;
      var windowSec = 30;
      var slidePct = 1 / windowSec * 100;
      var start = now - windowSec;
      var byDevice = {};
      for (var ri = 0; ri < requests.length; ri++) {
        var r = requests[ri];
        var ts = Number(r.arrived_at || 0);
        if (ts < start) continue;
        var did = r.device_id || "unknown";
        if (!byDevice[did]) byDevice[did] = { id: did, bars: [] };
        var leftPct = (ts - start) / windowSec * 100;
        var barWidth = Math.min(30, Math.max(1.5, Number(r.total_ms || 0) / 100));
        var status = r.status || "completed";
        byDevice[did].bars.push({ left: leftPct, width: barWidth, status: status, latency: Number(r.total_ms || 0), frame: r.frame_index, seq: r.sequence });
      }
      var devOrder = Object.keys(byDevice);
      var timeLabels = [{ pos: "0%", text: "30秒前" }, { pos: "50%", text: "15秒前" }, { pos: "100%", text: "现在" }];
      var xHtml = '<div class="heat-x-axis">';
      for (var ti = 0; ti < timeLabels.length; ti++) {
        var align = timeLabels[ti].pos === "100%" ? "translateX(-100%)" : (timeLabels[ti].pos === "0%" ? "translateX(0)" : "translateX(-50%)");
        xHtml += '<span style="left:' + timeLabels[ti].pos + ';transform:' + align + '">' + timeLabels[ti].text + '</span>';
      }
      xHtml += '</div>';
      if (devOrder.length === 0) {
        byId("streamTimeline").innerHTML = '<div class="empty">等待画面帧处理数据</div>';
        return;
      }
      var yHtml = '';
      var rowsHtml = '';
      for (var doi = 0; doi < devOrder.length; doi++) {
        yHtml += '<div class="heat-y-label">' + esc(friendlyDevice(devOrder[doi])) + '</div>';
        var bars = '';
        for (var bi = 0; bi < byDevice[devOrder[doi]].bars.length; bi++) {
          var b = byDevice[devOrder[doi]].bars[bi];
          bars += '<div class="stream-frame-bar sb-' + b.status + '" style="left:' + b.left.toFixed(2) + '%;width:' + b.width.toFixed(2) + '%" title="帧 #' + (b.seq || "-") + '（帧号 ' + (b.frame != null ? b.frame : "-") + '）· 耗时 ' + fmt(b.latency) + ' 毫秒 · ' + statusText(b.status) + '"></div>';
        }
        rowsHtml += '<div class="stream-dev-row"><div class="heat-slide" style="--slide-pct:' + slidePct.toFixed(3) + '%;--slide-dur:1s">' + bars + '</div></div>';
      }
      byId("streamTimeline").innerHTML = '<div class="heat-wrapper"><div class="heat-y-axis"><div class="heat-axis-spacer"></div>' + yHtml + '</div><div class="heat-main">' + xHtml + '<div class="stream-area">' + rowsHtml + '</div></div></div>';
    }

    function renderTaskStageGantt(snapshot) {
      var taskMap = {};
      var recent = snapshot.recent_requests || [];
      var running = snapshot.running_requests || [];
      for (var ri = 0; ri < recent.length; ri++) {
        if (recent[ri].request_id) taskMap[recent[ri].request_id] = recent[ri];
      }
      for (var rui = 0; rui < running.length; rui++) {
        if (running[rui].request_id) taskMap[running[rui].request_id] = Object.assign({}, taskMap[running[rui].request_id] || {}, running[rui]);
      }

      var stageMeta = {
        "Frame Upload": { label: "上传画面", cls: "stage-upload" },
        "Decode": { label: "解码", cls: "stage-decode" },
        "YOLO Detect": { label: "人物检测", cls: "stage-yolo" },
        "Track": { label: "目标跟踪", cls: "stage-track" },
        "ReID": { label: "身份比对", cls: "stage-track" },
        "YOLO-pose": { label: "姿态分析", cls: "stage-pose" },
        "State Fusion": { label: "状态识别", cls: "stage-state" },
        "Response": { label: "结果返回", cls: "stage-response" },
      };
      var deviceAlias = {};
      var devices = snapshot.devices || [];
      for (var di = 0; di < devices.length; di++) {
        deviceAlias[devices[di].device_id] = "用户" + (di + 1);
      }

      var nowSec = Date.now() / 1000;
      var windowSec = 20;
      var windowEnd = nowSec;
      var windowStart = windowEnd - windowSec;
      var slidePct = 1 / windowSec * 100;

      var bars = [];
      var values = Object.keys(taskMap).map(function (key) { return taskMap[key]; })
        .sort(function (a, b) { return Number(b.arrived_at || 0) - Number(a.arrived_at || 0); })
        .slice(0, 120);
      for (var ti = 0; ti < values.length; ti++) {
        var task = values[ti];
        var actual = task.actual_schedule || [];
        var planned = task.planned_schedule || [];
        var source = actual.length ? actual : planned;
        var sourceType = actual.length ? "actual" : "planned";
        for (var si = 0; si < source.length; si++) {
          var item = source[si];
          if (!stageMeta[item.stage]) continue;
          var start = Number(item.actual_start || item.planned_start || task.started_at || task.arrived_at || 0);
          var end = Number(item.actual_end || item.planned_end || 0);
          if (!end && item.duration_ms) end = start + Number(item.duration_ms) / 1000;
          if (!start || !end || end <= start) continue;
          bars.push({
            device_id: task.device_id || "unknown",
            sequence: task.sequence || "-",
            frame: task.frame_index,
            status: task.status || "queued",
            stage: item.stage,
            label: stageMeta[item.stage].label,
            cls: stageMeta[item.stage].cls,
            start: start,
            end: end,
            duration_ms: Number(item.duration_ms || (end - start) * 1000),
            type: sourceType,
          });
        }
      }

      if (!bars.length) {
        byId("taskStageGantt").innerHTML = '<div class="empty">等待用户上传画面后显示处理时段</div>';
        return;
      }

      var lanes = {};
      for (var bi = 0; bi < bars.length; bi++) {
        if (!lanes[bars[bi].device_id]) lanes[bars[bi].device_id] = [];
        lanes[bars[bi].device_id].push(bars[bi]);
      }
      var laneIds = Object.keys(lanes).slice(0, 8);
      var visibleBarsByDevice = {};
      for (var li = 0; li < laneIds.length; li++) {
        var deviceIdForWindow = laneIds[li];
        var selectedBars = lanes[deviceIdForWindow]
          .sort(function (a, b) { return Number(b.start || 0) - Number(a.start || 0); })
          .filter(function (bar) { return bar.end >= windowStart - 1 && bar.start <= windowEnd + 1; })
          .slice(0, 300)
          .sort(function (a, b) { return Number(a.start || 0) - Number(b.start || 0); });
        visibleBarsByDevice[deviceIdForWindow] = selectedBars;
      }
      var scale = '<div class="task-gantt-scale">' +
        '<span style="left:0%;transform:translateX(0)">20秒前</span>' +
        '<span style="left:50%;transform:translateX(-50%)">10秒前</span>' +
        '<span style="left:100%;transform:translateX(-100%)">现在</span>' +
        '</div>';
      var rows = laneIds.map(function (deviceId) {
        var deviceBars = visibleBarsByDevice[deviceId] || [];
        var alias = deviceAlias[deviceId] || ("用户" + (laneIds.indexOf(deviceId) + 1));
        var barsHtml = deviceBars.map(function (bar) {
          var left = Math.max(0, Math.min(99, (bar.start - windowStart) / windowSec * 100));
          var width = Math.max(1.2, Math.min(100 - left, (bar.end - bar.start) / windowSec * 100));
          var planClass = bar.type === "planned" ? " task-stage-plan" : "";
          var title = alias + " · 帧 #" + bar.sequence + "（帧号 " + (bar.frame ?? "-") + "）· " + bar.label + " · " + fmt(bar.duration_ms) + " 毫秒 · " + statusText(bar.status);
          return '<div class="task-stage-bar ' + bar.cls + planClass + '" style="left:' + left.toFixed(2) + '%;width:' + width.toFixed(2) + '%" title="' + esc(title) + '"></div>';
        }).join("");
        return '<div class="task-gantt-row">' +
          '<div class="task-gantt-device"><strong>' + esc(alias) + '</strong><span>' + esc(friendlyDevice(deviceId)) + '</span></div>' +
          '<div class="task-gantt-track"><div class="task-gantt-bars" style="--slide-pct:' + slidePct.toFixed(2) + '%;--slide-dur:1s">' + barsHtml + '</div></div>' +
          '</div>';
      }).join("");
      byId("taskStageGantt").innerHTML = '<div class="task-gantt">' + scale + rows + '</div>';
    }

    async function refresh() {
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" });
        const snapshot = await response.json();
        byId("syncTime").textContent = `已同步 ${new Date().toLocaleTimeString()}`;
        [
          renderCards,
          renderResources,
          renderDao,
          renderWorkflow,
          renderGantt,
          renderDevices,
          renderQueues,
          renderHistory,
          renderTaskStageGantt,
          renderDeviceHeatMap,
          renderStreamTimeline,
        ].forEach(fn => {
          try {
            fn(snapshot);
          } catch (error) {
            console.error(fn.name + " failed", error);
          }
        });
      } catch (error) {
        byId("syncTime").textContent = `同步失败 ${error.message}`;
      }
    }
    refresh();
    window.setInterval(refresh, 1000);

    // ── 一级标签页切换（参考 Challenge-cup 管理端风格） ──
    document.querySelectorAll(".tabs .tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tabs .tab").forEach(item => item.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.view;
        document.querySelectorAll(".view").forEach(view => {
          view.classList.toggle("active", view.id === target);
        });
      });
    });
    // ── 甘特图子栏：三图自由勾选 ──
    document.querySelectorAll(".chart-toggles input[data-chart]").forEach(box => {
      box.addEventListener("change", () => {
        const target = document.getElementById(box.dataset.chart);
        if (target) target.style.display = box.checked ? "" : "none";
      });
    });
  </script>
</body>
</html>"""


def start_server_admin(service: InferenceServiceImpl, port: int):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/snapshot"):
                self._send_json(service.monitor_snapshot())
                return
            data = SERVER_ADMIN_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002
            logger.debug("admin http: " + format, *args)

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, name="server-admin", daemon=True)
    thread.start()
    return httpd, thread


def serve(port: int, yolo_model: str, osnet_model: str | None, device: str,
          extra_latency_ms: float = 0.0, max_workers: int = 4,
          movenet_model: str = "data/models/movenet_lightning.onnx",
          torch_device: str = "auto",
          enable_reid: bool = False,
          enable_education: bool = False,
          face_model: str | None = None,
          expression_model: str | None = None,
          head_pose_model: str | None = None,
          gaze_model: str | None = None,
          education_csv: str | None = None,
          face_landmarker_model: str | None = None,
          monitor_url: str | None = None,
          admin_port: int | None = None,
          gpu_stress: bool = False,
          gpu_stress_streams: int = 4,
          gpu_stress_fps: float = 8.0,
          gpu_stress_frames: int = 64,
          gpu_stress_width: int = 640,
          gpu_stress_height: int = 360,
          gpu_stress_video: str | None = None,
          gpu_stress_duration: float = 0.0):
    # gRPC 线程池要足够大，让所有请求都能进入 handler
    # 真正的推理并发由 InferenceServiceImpl 内部的 _infer_sem 控制
    grpc_threads = max(20, max_workers * 4)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_threads),
        options=[
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),  # 10 MB
            ("grpc.max_send_message_length", 10 * 1024 * 1024),
            # keepalive：允许客户端在空闲时发 ping 保活（配合树莓派端 keepalive 设置）
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_time_between_pings_ms", 10000),  # 允许最少 10s 一次 ping
            ("grpc.http2.max_pings_burst", 10),               # 允许突发 10 个 ping
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ],
    )
    service = InferenceServiceImpl(
        yolo_model,
        osnet_model,
        device,
        extra_latency_ms=extra_latency_ms,
        max_workers=max_workers,
        movenet_model=movenet_model,
        torch_device=torch_device,
        enable_reid=enable_reid,
        enable_education=enable_education,
        face_model=face_model,
        expression_model=expression_model,
        head_pose_model=head_pose_model,
        gaze_model=gaze_model,
        education_csv=education_csv,
        face_landmarker_model=face_landmarker_model,
    )
    pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    reporter_stop_event = None
    reporter_thread = None
    admin_httpd = None
    admin_thread = None
    stress_stop_event = None
    stress_threads: list[threading.Thread] = []
    if monitor_url:
        reporter_stop_event, reporter_thread = start_monitor_reporter(service, monitor_url)
        logger.info(f"运行态监控上报已启动: {monitor_url.rstrip('/')}/ingest/server")
    if admin_port:
        admin_httpd, admin_thread = start_server_admin(service, admin_port)
        logger.info(f"边缘服务器 DAO 管理端已启动: http://0.0.0.0:{admin_port}/")

    # 获取本机 IP
    if gpu_stress:
        stress_stop_event, stress_threads = start_gpu_stress(
            service=service,
            streams=gpu_stress_streams,
            fps=gpu_stress_fps,
            frame_count=gpu_stress_frames,
            width=gpu_stress_width,
            height=gpu_stress_height,
            video_path=gpu_stress_video,
            duration_sec=gpu_stress_duration,
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("\ngRPC 推理服务已启动")
    logger.info(f"  地址: {local_ip}:{port}")
    logger.info(f"  设备: {device}")
    logger.info(f"  推理并发: max_workers={max_workers} (gRPC threads={grpc_threads})")
    logger.info("  等待树莓派、安卓手机等边缘设备连接...\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("服务关闭")
    finally:
        if reporter_stop_event is not None:
            reporter_stop_event.set()
        if reporter_thread is not None:
            reporter_thread.join(timeout=1.0)
        if admin_httpd is not None:
            admin_httpd.shutdown()
        if admin_thread is not None:
            admin_thread.join(timeout=1.0)
        if stress_stop_event is not None:
            stress_stop_event.set()
        for thread in stress_threads:
            thread.join(timeout=1.0)
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description="电脑边缘服务器 gRPC 推理服务")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO 检测模型，默认与前端一致使用 Ultralytics .pt")
    parser.add_argument("--osnet-model", default="data/models/osnet_x0_25_slim.onnx")
    parser.add_argument(
        "--torch-device",
        default="auto",
        help="Device for Ultralytics .pt models and State Fusion: auto, cuda, cuda:0, or cpu",
    )
    parser.add_argument("--enable-reid", action="store_true", help="Enable OSNet/ReID model loading")
    parser.add_argument("--disable-reid", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--extra-latency-ms", type=float, default=0.0,
        help="模拟弱网: 每个请求额外延迟 (ms), 上下行各一半",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="gRPC 线程池大小 (限制并发处理能力)",
    )
    parser.add_argument(
        "--movenet-model", default="yolov8n-pose.pt",
        help="姿态模型路径，默认与前端一致使用 yolov8n-pose.pt；也兼容 MoveNet ONNX",
    )
    parser.add_argument(
        "--enable-education",
        action="store_true",
        default=True,
        help="Enable YOLO + State Fusion education-state inference (enabled by default)",
    )
    parser.add_argument(
        "--disable-education",
        action="store_true",
        help="Disable YOLO + State Fusion education-state inference and return detections only",
    )
    parser.add_argument(
        "--face-model",
        default="data/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml",
        help="Face detector (OpenVINO .xml or ONNX); defaults to OpenVINO ADAS model",
    )
    parser.add_argument(
        "--expression-model",
        default="data/models/intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml",
        help="Expression classifier (OpenVINO .xml or ONNX); defaults to OpenVINO emotions-recognition-retail-0003",
    )
    parser.add_argument(
        "--head-pose-model",
        default="data/models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml",
        help="Head-pose estimator (OpenVINO .xml or ONNX); defaults to OpenVINO ADAS model",
    )
    parser.add_argument(
        "--gaze-model",
        default="data/models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml",
        help="Gaze estimator (OpenVINO .xml or ONNX); defaults to OpenVINO ADAS model",
    )
    parser.add_argument(
        "--face-landmarker-model",
        default="data/models/mediapipe/face_landmarker.task",
        help="MediaPipe FaceLandmarker .task model for EAR-based eye-closure detection",
    )
    parser.add_argument(
        "--education-csv",
        default=None,
        help="Path to save per-frame education metrics CSV. Defaults to data/results/education_runtime_<timestamp>.csv",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="运行模式：启动 mDNS 广播，供安卓手机等边缘设备自动发现",
    )
    parser.add_argument(
        "--service-name",
        default="edge-srv",
        help="运行模式 mDNS 服务名",
    )
    parser.add_argument(
        "--streams-max",
        type=int,
        default=3,
        help="运行模式广播使用的最大流数",
    )
    parser.add_argument(
        "--monitor-url",
        default=None,
        help="集中监控服务地址，例如 http://127.0.0.1:8088",
    )
    parser.add_argument(
        "--admin-port",
        type=int,
        default=8011,
        help="Windows 边缘服务器 DAO 调度管理端 HTTP 端口；设为 0 可关闭",
    )
    parser.add_argument(
        "--gpu-stress",
        action="store_true",
        help="Run local continuous offload requests to make GPU utilization visible",
    )
    parser.add_argument("--gpu-stress-streams", type=int, default=4)
    parser.add_argument("--gpu-stress-fps", type=float, default=8.0)
    parser.add_argument("--gpu-stress-frames", type=int, default=64)
    parser.add_argument("--gpu-stress-width", type=int, default=640)
    parser.add_argument("--gpu-stress-height", type=int, default=360)
    parser.add_argument("--gpu-stress-video", default=None)
    parser.add_argument(
        "--gpu-stress-duration",
        type=float,
        default=0.0,
        help="Stop GPU stress after N seconds; 0 means run until server stops",
    )
    args = parser.parse_args()

    announcer = None
    if args.runtime:
        announcer = RuntimeAnnouncer(
            port=args.port,
            service_name=args.service_name,
            props={
                "version": "0.4",
                "mode": "runtime",
                "gpu": "cuda" if args.device == "gpu" else "cpu",
                "streams_max": str(args.streams_max),
            },
        )
        announcer.start()

    try:
        serve(
            args.port,
            args.yolo_model,
            args.osnet_model,
            args.device,
            extra_latency_ms=args.extra_latency_ms,
            max_workers=args.max_workers,
            movenet_model=args.movenet_model,
            torch_device=args.torch_device,
            enable_reid=args.enable_reid and not args.disable_reid,
            enable_education=args.enable_education and not args.disable_education,
            face_model=args.face_model,
            expression_model=args.expression_model,
            head_pose_model=args.head_pose_model,
            gaze_model=args.gaze_model,
            education_csv=args.education_csv,
            face_landmarker_model=args.face_landmarker_model,
            monitor_url=args.monitor_url,
            admin_port=args.admin_port or None,
            gpu_stress=args.gpu_stress,
            gpu_stress_streams=args.gpu_stress_streams,
            gpu_stress_fps=args.gpu_stress_fps,
            gpu_stress_frames=args.gpu_stress_frames,
            gpu_stress_width=args.gpu_stress_width,
            gpu_stress_height=args.gpu_stress_height,
            gpu_stress_video=args.gpu_stress_video,
            gpu_stress_duration=args.gpu_stress_duration,
        )
    finally:
        if announcer is not None:
            announcer.stop()


if __name__ == "__main__":
    main()
