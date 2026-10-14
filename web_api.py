"""
FastAPI + WebSocket entry for the AI virtual teacher demo.

This keeps the original Streamlit app as a fallback while exposing a more
product-like Vue frontend from the same Python backend.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Literal
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart_education import DeviceProfiler, SmartEducationOffload


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "web" / "vue-teacher"
ADMIN_FRONTEND_DIR = BASE_DIR / "web" / "dao-admin"
LESSON_DIR = BASE_DIR / "data" / "lessons"
STUDENT_VIDEO_DIR = BASE_DIR / "data" / "videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
LLM_CONFIG_PATH = BASE_DIR / "llm_config.local.json"
MAX_BROWSER_CAMERA_FRAME_BYTES = int(
    os.getenv("BROWSER_CAMERA_MAX_FRAME_BYTES", str(2 * 1024 * 1024))
)

LESSON_CATALOG = [
    {
        "id": "songhao-01-mapping",
        "title": "宋浩高等数学：1 映射",
        "teacher": "宋浩老师",
        "subject": "高等数学",
        "difficulty": "基础",
        "knowledge": "映射、定义域、值域、对应关系",
        "pace": "标准",
        "local_video": "1 映射.mp4",
        "source_url": "https://www.bilibili.com/video/BV1UW411k7Jv/",
        "content": [
            "映射强调从一个集合到另一个集合的确定对应关系。",
            "理解定义域和值域是学习函数概念的基础。",
            "函数可以看作一种特殊的映射。",
        ],
        "quiz": "映射和函数之间有什么关系？",
    },
    {
        "id": "songhao-02-function",
        "title": "宋浩高等数学：2 函数",
        "teacher": "宋浩老师",
        "subject": "高等数学",
        "difficulty": "基础",
        "knowledge": "函数定义、定义域、值域、函数图像",
        "pace": "标准",
        "local_video": "2 函数.mp4",
        "source_url": "https://www.bilibili.com/video/BV1UW411k7Jv/",
        "content": [
            "函数描述两个变量之间的对应关系。",
            "判断函数时要先看定义域和值域。",
            "函数可以用解析式、表格和图像表示。",
        ],
        "quiz": "为什么研究函数时要先确定定义域？",
    },
    {
        "id": "songhao-03-elementary-function",
        "title": "宋浩高等数学：3 初等函数",
        "teacher": "宋浩老师",
        "subject": "高等数学",
        "difficulty": "基础",
        "knowledge": "基本初等函数、复合函数、初等函数",
        "pace": "标准",
        "local_video": "3 初等函数.mp4",
        "source_url": "https://www.bilibili.com/video/BV1UW411k7Jv/",
        "content": [
            "基本初等函数包括幂函数、指数函数、对数函数、三角函数等。",
            "复合函数由多个函数按内外层关系组合而成。",
            "初等函数是高等数学中求极限、求导和积分的重要对象。",
        ],
        "quiz": "复合函数和基本初等函数之间有什么关系？",
    },
    {
        "id": "songhao-04-sequence-limit",
        "title": "宋浩高等数学：4 数列的极限",
        "teacher": "宋浩老师",
        "subject": "高等数学",
        "difficulty": "基础",
        "knowledge": "数列极限、收敛、趋近思想",
        "pace": "慢速",
        "local_video": "4 数列的极限.mp4",
        "source_url": "https://www.bilibili.com/video/BV1UW411k7Jv/",
        "content": [
            "数列极限描述项数无限增大时数列项的变化趋势。",
            "数列收敛意味着数列项逐渐接近某个确定常数。",
            "理解趋近过程是学习函数极限和连续性的基础。",
        ],
        "quiz": "数列极限中的“趋近”是什么意思？",
    },
    {
        "id": "songhao-05-convergent-sequence",
        "title": "宋浩高等数学：5 收敛数列的性质",
        "teacher": "宋浩老师",
        "subject": "高等数学",
        "difficulty": "基础",
        "knowledge": "收敛数列、有界性、唯一性、保号性",
        "pace": "标准",
        "local_video": "5 收敛数列的性质.mp4",
        "source_url": "https://www.bilibili.com/video/BV1UW411k7Jv/",
        "content": [
            "收敛数列的极限具有唯一性。",
            "收敛数列一定有界。",
            "保号性常用于判断极限附近项的正负关系。",
        ],
        "quiz": "为什么说收敛数列一定有界？",
    },
]

LESSON_TIMELINE = {
    "songhao-01-mapping": [
        {"start": 178, "end": 365, "topic": "映射、像和原像、定义域和值域的定义"},
        {"start": 368, "end": 883, "topic": "像和原像的关系，像是唯一的，原像不唯一"},
        {"start": 885, "end": 1090, "topic": "满射、单射与一一映射"},
        {"start": 1093, "end": 1370, "topic": "逆映射的定义，只有单射才有逆映射"},
        {"start": 1383, "end": 1765, "topic": "复合映射定义"},
        {"start": 1767, "end": 2065, "topic": "复合映射案例 sqrt(1 - sin^2 x) 和 sqrt(-1 - 4x^2)"},
    ],
    "songhao-02-function": [
        {"start": 0, "end": 250, "topic": "函数的定义和要素"},
        {"start": 250, "end": 716, "topic": "函数的案例"},
        {"start": 720, "end": 2619, "topic": "函数的有界性、单调性、奇偶性、周期性"},
        {"start": 2620, "end": 3300, "topic": "反函数的定义、案例和求反函数"},
        {"start": 3300, "end": None, "topic": "复合函数的定义和案例"},
    ],
    "songhao-03-elementary-function": [
        {"start": 26, "end": 73, "topic": "幂函数定义、图像和性质"},
        {"start": 74, "end": 158, "topic": "指数函数和 e^x 定义、图像和性质"},
        {"start": 159, "end": 338, "topic": "对数函数和 ln x 定义、图像和性质"},
        {"start": 339, "end": 1018, "topic": "三角函数 sin x、cos x、tan x、cot x、sec x、csc x 定义、图像和性质"},
        {"start": 1020, "end": 1808, "topic": "反三角函数 arcsin x、arccos x、arctan x、arccot x 定义、图像和性质"},
    ],
    "songhao-04-sequence-limit": [
        {"start": 46, "end": 116, "topic": "数列概念、无穷项、通项"},
        {"start": 117, "end": 204, "topic": "数列的符号"},
        {"start": 205, "end": 968, "topic": "数列案例 {1/n} 引入数列极限的定义"},
        {"start": 970, "end": 1355, "topic": "数列极限定义"},
        {"start": 1365, "end": 1869, "topic": "数列 {1/n} 的极限证明"},
        {"start": 1879, "end": 2277, "topic": "数列 (n + (-1)^(n-1)) / n 的极限证明"},
        {"start": 2288, "end": 2779, "topic": "数列 (-1)^n / (n + 1)^2 的极限证明"},
        {"start": 2780, "end": 3370, "topic": "|q| < 1 时数列 q^n 的极限证明"},
    ],
    "songhao-05-convergent-sequence": [
        {"start": 0, "end": 450, "topic": "数列极限的唯一性"},
        {"start": 453, "end": 902, "topic": "数列极限的有界性"},
        {"start": 903, "end": 1201, "topic": "数列极限保号性"},
        {"start": 1202, "end": 1582, "topic": "子数列的收敛与发散，以及子数列极限和原数列极限的关系"},
    ],
}

STATE_LABELS = {
    "focused": "专注",
    "confused": "困惑",
    "tired": "疲劳",
    "distracted": "分心",
    "absent": "缺勤",
    "unknown": "未知",
    "offline": "离线",
}

STATE_ORDER = ["focused", "confused", "tired", "distracted", "absent"]

ANALYSIS_MODE_CONFIG = {
    "smooth": {"label": "流畅模式", "pose_interval": 5},
    "standard": {"label": "标准模式", "pose_interval": 3},
    "accurate": {"label": "高精度模式", "pose_interval": 1},
}


def normalize_learning_state(state: str | None, has_detection: bool = True) -> str:
    if state in STATE_ORDER:
        return state
    return "focused" if has_detection else "absent"


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def resolve_lesson_segment(lesson_id: str, playback_time: float | None) -> dict | None:
    if playback_time is None:
        return None
    try:
        current = max(0.0, float(playback_time))
    except (TypeError, ValueError):
        return None
    for item in LESSON_TIMELINE.get(lesson_id, []):
        end = item.get("end")
        if current >= float(item["start"]) and (end is None or current <= float(end)):
            return {
                **item,
                "playback_time": round(current, 1),
                "time_label": f"{format_seconds(item['start'])}-{format_seconds(end)}",
            }
    return {
        "start": None,
        "end": None,
        "topic": "当前播放位置未标注知识点",
        "playback_time": round(current, 1),
        "time_label": format_seconds(current),
    }


def lesson_with_segment(lesson: dict, segment: dict | None) -> dict:
    if not segment:
        return lesson
    topic = segment["topic"]
    return {
        **lesson,
        "current_segment": segment,
        "knowledge": topic,
        "content": [
            f"当前片段：{segment['time_label']}",
            topic,
            *lesson.get("content", [])[:2],
        ],
        "quiz": f"请用自己的话概括：{topic} 的核心思想是什么？",
    }

def load_local_llm_config() -> dict:
    if not LLM_CONFIG_PATH.exists():
        return {}
    try:
        with LLM_CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


local_llm_config = load_local_llm_config()

settings = {
    "edge_host": os.getenv("EDGE_HOST", "192.168.1.104"),
    "edge_port": 50051,
    "use_edge": os.getenv("EDGE_USE_EDGE", "").lower() in {"1", "true", "yes", "on"},
    "student_video_dir": str(STUDENT_VIDEO_DIR),
    "camera_id": 0,
    "analysis_width": int(os.getenv("ANALYSIS_WIDTH", "640")),
    "analysis_height": int(os.getenv("ANALYSIS_HEIGHT", "360")),
    "llm_enabled": (
        os.getenv("VIRTUAL_TEACHER_LLM_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        or bool(local_llm_config.get("enabled"))
    ),
    "llm_api_key": os.getenv("LLM_API_KEY") or local_llm_config.get("api_key", ""),
    "llm_api_base": (
        os.getenv("LLM_API_BASE")
        or local_llm_config.get("api_base")
        or "https://api.deepseek.com"
    ).rstrip("/"),
    "llm_model": os.getenv("LLM_MODEL") or local_llm_config.get("model", "deepseek-chat"),
    "llm_timeout_seconds": float(
        os.getenv("LLM_TIMEOUT_SECONDS") or local_llm_config.get("timeout_seconds", 30)
    ),
    # ---- 线下课堂管理端上报配置 ----
    "classroom_dashboard_url": os.getenv("CLASSROOM_DASHBOARD_URL", ""),
    "seat_label": os.getenv("SEAT_LABEL", ""),
    "student_name": os.getenv("STUDENT_NAME", ""),
}


def normalize_client_id(raw_device_id: str | None) -> str:
    host = platform.node() or "edge-device"
    raw = (raw_device_id or "").strip() or "user-default"
    safe_user = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw)[:48]
    safe_host = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in host)[:48]
    if safe_user.startswith("client_") or "__user-" in safe_user:
        return safe_user
    return f"client_{safe_host}__{safe_user}"


analyzer: SmartEducationOffload | None = None
analyzer_lock = threading.Lock()
feedback_cache: dict[str, dict] = {}
last_teaching_feedback: dict[str, object] = {}
last_non_absent_report: dict | None = None
consecutive_empty_frames = 0
MAX_SCHEDULER_EVENTS = 120
scheduler_events: list[dict] = []

# ---- 线下课堂管理端上报状态 ----
_classroom_latest_reports: dict[str, dict] = {}
_classroom_report_lock = threading.Lock()
_classroom_report_thread: threading.Thread | None = None

app = FastAPI(title="AI Virtual Teacher Demo", version="0.2.0")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/admin-assets", StaticFiles(directory=ADMIN_FRONTEND_DIR), name="admin_assets")
app.mount("/media/lessons", StaticFiles(directory=LESSON_DIR), name="lesson_media")

CLASSROOM_CLIENT_DIR = BASE_DIR / "web" / "classroom-client"
if CLASSROOM_CLIENT_DIR.exists():
    app.mount("/client-assets", StaticFiles(directory=CLASSROOM_CLIENT_DIR), name="client_assets")

@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    response = await call_next(request)
    if (
        request.url.path in {"/", "/admin", "/client"}
        or request.url.path.startswith("/assets/")
        or request.url.path.startswith("/admin-assets/")
        or request.url.path.startswith("/client-assets/")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.on_event("startup")
async def warmup_analyzer_on_startup():
    # YOLO / pose model loading is slow on Windows. Warm it up in the background
    # so the first classroom click does not look like a stalled decision.
    if settings["use_edge"]:
        asyncio.create_task(asyncio.to_thread(get_analyzer))
    # 启动线下课堂管理端上报线程
    global _classroom_report_thread
    if _classroom_report_thread is None or not _classroom_report_thread.is_alive():
        _classroom_report_thread = threading.Thread(
            target=_classroom_report_loop, daemon=True
        )
        _classroom_report_thread.start()


class AnalyzeRequest(BaseModel):
    source: Literal["camera", "sample"] = "sample"
    sample_video: str | None = None
    device_id: str | None = None


class SettingsRequest(BaseModel):
    edge_host: str | None = None
    edge_port: int | None = None
    use_edge: bool | None = None
    camera_id: int | None = None
    student_video_dir: str | None = None
    llm_timeout_seconds: float | None = None
    classroom_dashboard_url: str | None = None
    seat_label: str | None = None
    student_name: str | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None
    api_base: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    lesson_id: str | None = None


def get_analyzer() -> SmartEducationOffload:
    global analyzer
    with analyzer_lock:
        if analyzer is None:
            analyzer = SmartEducationOffload(
                edge_host=settings["edge_host"] if settings["use_edge"] else None,
                edge_port=settings["edge_port"] if settings["use_edge"] else None,
            )
    return analyzer


def reset_analyzer() -> None:
    global analyzer
    with analyzer_lock:
        analyzer = None
    last_teaching_feedback.clear()


def reset_presence_memory() -> None:
    global last_non_absent_report, consecutive_empty_frames
    last_non_absent_report = None
    consecutive_empty_frames = 0


def public_settings() -> dict:
    return {
        key: value
        for key, value in settings.items()
        if key != "llm_api_key"
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _now_label(timestamp: float | None = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))


def _dominant_state(state_counts: dict) -> str:
    if not state_counts:
        return "absent"
    key, value = max(state_counts.items(), key=lambda item: item[1] or 0)
    return key if value else "absent"


def _current_dao_info() -> dict:
    if analyzer is not None:
        dao = getattr(analyzer, "dao_decision_info", None)
        if isinstance(dao, dict):
            return dao
    if scheduler_events:
        dao = scheduler_events[-1].get("dao")
        if isinstance(dao, dict):
            return dao
    return {
        "enabled": True,
        "policy": "DaoPolicy",
        "strategy": "waiting",
        "admission": "waiting",
        "reason": "等待边缘设备提交视频分析任务。",
        "priority": 0,
        "stage_count": 5,
        "stages_to_offload": [],
    }


def current_edge_payload(strategy: str | None = None) -> dict:
    dao = _current_dao_info()
    connected = bool(getattr(analyzer, "edge_connected", False) if analyzer else False)
    current_strategy = (
        strategy
        or getattr(analyzer, "strategy", None)
        or dao.get("strategy")
        or "waiting"
    )
    return {
        "strategy": current_strategy,
        "connected": connected,
        "dao": dao,
    }


def _stage_is_offloaded(stage_id: str, stages_to_offload: list, strategy: str) -> bool:
    if strategy != "offload":
        return False
    if not stages_to_offload:
        return stage_id in {"detect", "pose", "state"}
    stage_text = " ".join(str(item).lower() for item in stages_to_offload)
    aliases = {
        "sample": ["decode", "sample", "frame"],
        "detect": ["detect", "yolo"],
        "pose": ["pose", "behavior"],
        "state": ["state", "fusion", "classify"],
        "teacher": ["postprocess", "teacher", "feedback"],
    }
    return any(alias in stage_text for alias in aliases.get(stage_id, [stage_id]))


def _pipeline_from_dao(dao: dict) -> list[dict]:
    strategy = str(dao.get("strategy", "waiting"))
    stages_to_offload = dao.get("stages_to_offload") or []
    if not isinstance(stages_to_offload, list):
        stages_to_offload = []
    base_stages = [
        {
            "id": "sample",
            "name": "视频采样",
            "model": "视频采集与传输",
            "local": "树莓派/安卓端",
            "edge": "边缘设备",
        },
        {
            "id": "detect",
            "name": "人体/手机检测",
            "model": "人物检测",
            "local": "树莓派 CPU",
            "edge": "Windows 边缘服务器 GPU/CPU",
        },
        {
            "id": "pose",
            "name": "姿态与行为特征",
            "model": "姿态检查",
            "local": "树莓派 CPU",
            "edge": "Windows 边缘服务器 GPU/CPU",
        },
        {
            "id": "state",
            "name": "学习状态分类",
            "model": "表情与课堂状态识别",
            "local": "树莓派 CPU",
            "edge": "Windows 边缘服务器 GPU/CPU",
        },
        {
            "id": "teacher",
            "name": "AI 教师干预",
            "model": "规则/大模型反馈",
            "local": "客户端展示",
            "edge": "边缘服务器/云端 API",
        },
    ]
    pipeline = []
    for stage in base_stages:
        offloaded = _stage_is_offloaded(stage["id"], stages_to_offload, strategy)
        pipeline.append({
            **stage,
            "placement": stage["edge"] if offloaded else stage["local"],
            "offloaded": offloaded,
        })
    return pipeline


def record_scheduler_event(payload: dict, device_id: str | None = None, source: str | None = None) -> None:
    report = payload.get("report") or {}
    edge = payload.get("edge") or {}
    dao = edge.get("dao") or {}
    if not isinstance(dao, dict):
        dao = {}
    sample = payload.get("sample") or {}
    lesson_context = payload.get("lesson_context") or {}
    state_counts = report.get("state_counts") or {}
    timestamp = _safe_float(payload.get("timestamp"), time.time())
    strategy = str(edge.get("strategy") or dao.get("strategy") or report.get("strategy") or "unknown")
    latency_ms = _safe_float(payload.get("elapsed_ms"), _safe_float(report.get("total_ms"), 0.0))
    event = {
        "id": f"{int(timestamp * 1000)}-{len(scheduler_events) % MAX_SCHEDULER_EVENTS}",
        "timestamp": timestamp,
        "time_label": _now_label(timestamp),
        "device_id": device_id or "rpi-client-01",
        "device_name": "树莓派边缘设备" if (device_id or "").startswith("rpi") else "边缘客户端",
        "source": source or sample.get("source") or "sample",
        "lesson_title": lesson_context.get("title", "宋浩高等数学"),
        "frame_index": sample.get("frame_index"),
        "analysis_mode": sample.get("analysis_mode") or report.get("analysis_mode") or "standard",
        "strategy": strategy,
        "admission": dao.get("admission") or ("accept" if strategy == "offload" else "local"),
        "reason": dao.get("reason", ""),
        "priority": _safe_float(dao.get("priority"), 0.0),
        "target_node_id": dao.get("target_node_id") or "windows-edge-server",
        "stages_to_offload": dao.get("stages_to_offload") or [],
        "latency_ms": round(latency_ms, 1),
        "edge_latency_ms": round(_safe_float(dao.get("edge_latency_ms"), 0.0), 1),
        "transfer_ms": round(_safe_float(dao.get("estimated_transfer_ms"), 0.0), 1),
        "deadline_ms": round(_safe_float(dao.get("deadline_ms"), 0.0), 1),
        "queue": _safe_int(dao.get("edge_queue"), 0),
        "local_cpu": round(_safe_float(dao.get("local_cpu"), 0.0), 1),
        "edge_cpu": round(_safe_float(dao.get("edge_cpu"), 0.0), 1),
        "edge_gpu": round(_safe_float(dao.get("edge_gpu"), 0.0), 1),
        "connected": bool(edge.get("connected", False)),
        "detections": _safe_int(report.get("detections"), 0),
        "student_total": _safe_int(report.get("total"), 0),
        "state": _dominant_state(state_counts),
        "state_label": STATE_LABELS.get(_dominant_state(state_counts), _dominant_state(state_counts)),
        "state_counts": state_counts,
        "dao": dao,
    }
    scheduler_events.append(event)
    if len(scheduler_events) > MAX_SCHEDULER_EVENTS:
        del scheduler_events[: len(scheduler_events) - MAX_SCHEDULER_EVENTS]


def admin_snapshot_payload() -> dict:
    events = list(scheduler_events)
    recent_events = events[-40:]
    latest = events[-1] if events else {}
    dao = _current_dao_info()
    strategy = str(dao.get("strategy") or latest.get("strategy") or "waiting")
    offload_count = sum(1 for item in events if item.get("strategy") == "offload")
    local_count = sum(1 for item in events if item.get("strategy") == "local")
    accepted_count = sum(1 for item in events if item.get("admission") == "accept")
    avg_latency = (
        sum(_safe_float(item.get("latency_ms")) for item in recent_events) / len(recent_events)
        if recent_events
        else 0.0
    )
    recent_device_ids = {
        item.get("device_id")
        for item in events
        if time.time() - _safe_float(item.get("timestamp"), 0) < 30
    }
    edge_connected = bool(latest.get("connected")) or bool(getattr(analyzer, "edge_connected", False) if analyzer else False)
    device_cpu = DeviceProfiler.get_cpu_percent(interval=0.05)
    device_memory = DeviceProfiler.get_memory_percent()
    device_score = DeviceProfiler.estimate_performance_score()
    def device_task_count(device_id: str) -> int:
        return sum(1 for item in events if item.get("device_id") == device_id)

    def device_last_event(device_id: str) -> dict:
        return next((item for item in reversed(events) if item.get("device_id") == device_id), {})

    device_ids = sorted({str(item.get("device_id")) for item in events if item.get("device_id")})
    classroom_devices = []
    for device_id in device_ids:
        last = device_last_event(device_id)
        short_id = device_id.replace("client_", "", 1).split("__", 1)[0]
        classroom_devices.append(
            {
                "id": device_id,
                "name": f"用户 {short_id}",
                "role": "课堂采集端",
                "status": "active" if device_id in recent_device_ids else "standby",
                "tasks": device_task_count(device_id),
                "last_seen": last.get("time_label", "--"),
                "strategy": last.get("strategy", "waiting"),
                "state": last.get("state_label", "等待采样"),
                "latency_ms": last.get("latency_ms", 0),
            }
        )
    offload_ratio = round((offload_count / len(events) * 100) if events else 0.0, 1)
    slo_ms = _safe_float(dao.get("deadline_ms"), 200.0) or 200.0
    edge_latency = _safe_float(latest.get("edge_latency_ms"), _safe_float(dao.get("edge_latency_ms"), 0.0))
    transfer_ms = _safe_float(latest.get("transfer_ms"), _safe_float(dao.get("estimated_transfer_ms"), 0.0))
    local_baseline = max(avg_latency * 1.35, avg_latency + 80) if avg_latency else 0.0
    offload_estimate = max(avg_latency, edge_latency + transfer_ms) if avg_latency else 0.0
    return {
        "timestamp": time.time(),
        "summary": {
            "total_requests": len(events),
            "offload_count": offload_count,
            "local_count": local_count,
            "accepted_count": accepted_count,
            "rejected_count": max(0, len(events) - accepted_count - local_count),
            "offload_ratio": offload_ratio,
            "avg_latency_ms": round(avg_latency, 1),
            "slo_ms": round(slo_ms, 1),
            "slo_pass": bool(avg_latency and avg_latency <= slo_ms),
            "online_devices": sum(1 for item in classroom_devices if item["status"] == "active"),
        },
        "server": {
            "id": "windows-edge-server",
            "name": "Windows 边缘服务器",
            "host": settings["edge_host"],
            "port": settings["edge_port"],
            "use_edge": settings["use_edge"],
            "connected": edge_connected,
            "cpu": round(_safe_float(dao.get("edge_cpu"), device_cpu), 1),
            "gpu": round(_safe_float(dao.get("edge_gpu"), 0.0), 1),
            "memory": round(device_memory, 1),
            "local_score": round(_safe_float(dao.get("local_score"), device_score), 1),
            "queue": _safe_int(dao.get("edge_queue"), _safe_int(latest.get("queue"), 0)),
            "edge_latency_ms": round(edge_latency, 1),
        },
        "dao": {
            **dao,
            "strategy": strategy,
            "admission": dao.get("admission") or latest.get("admission") or "waiting",
            "reason": dao.get("reason") or latest.get("reason") or "等待调度决策。",
        },
        "devices": classroom_devices,
        "pipeline": _pipeline_from_dao({**dao, "strategy": strategy}),
        "comparison": {
            "local_baseline_ms": round(local_baseline, 1),
            "offload_estimate_ms": round(offload_estimate, 1),
            "saved_ms": round(max(0.0, local_baseline - offload_estimate), 1),
        },
        "events": list(reversed(recent_events)),
    }


def classroom_admin_snapshot(timeout: float = 2.5) -> dict | None:
    """读取教师管理端汇总的全课堂调度快照。

    8009 可能运行在任意一台学生设备上，本进程只能看到自己的任务；8008 会把
    8011 边缘服务器记录的所有学生设备统一适配成管理页结构。连接失败时返回
    None，由调用方退回本机快照，保证单机演示仍可使用。
    """
    # 启动脚本/树莓派传入的环境变量必须优先，避免旧的本地设置把管理页代理回
    # 127.0.0.1:8008，导致每台树莓派只能看到自己或错误的服务器。
    base_url = str(
        os.getenv("CLASSROOM_DASHBOARD_URL", "").strip()
        or settings.get("classroom_dashboard_url")
        or "http://127.0.0.1:8008"
    ).strip().rstrip("/")
    if not base_url:
        return None
    try:
        with urlrequest.urlopen(
            base_url + "/api/admin/dao-snapshot", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def lesson_payload(lesson: dict) -> dict:
    video_path = LESSON_DIR / lesson["local_video"]
    return {
        **lesson,
        "video_url": f"/api/lesson-video/{lesson['id']}",
        "video_exists": video_path.exists(),
    }


def find_lesson(lesson_id: str) -> dict:
    return next((item for item in LESSON_CATALOG if item["id"] == lesson_id), LESSON_CATALOG[0])


def list_student_videos() -> list[dict]:
    root = Path(settings["student_video_dir"]).expanduser()
    if not root.is_absolute():
        root = BASE_DIR / root
    if not root.exists():
        return []
    videos = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append({"name": path.name, "relative_path": str(path.relative_to(root))})
    return videos


def resolve_student_video(relative_path: str | None) -> Path | None:
    if not relative_path:
        videos = list_student_videos()
        if not videos:
            return None
        relative_path = videos[0]["relative_path"]
    root = Path(settings["student_video_dir"]).expanduser()
    if not root.is_absolute():
        root = BASE_DIR / root
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Invalid video path")
    return candidate


def read_sample_frame(relative_path: str | None):
    video_path = resolve_student_video(relative_path)
    if video_path is None or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _open_camera_capture(camera_id: int):
    """Open a camera capture handle, preferring DirectShow on Windows.

    OpenCV 在 Windows 上默认使用 MSMF 后端，部分机型会出现打开极慢（单次数秒）
    甚至打不开的问题；DirectShow 后端更快且兼容性更好，故优先尝试，
    失败后再回退到默认后端。
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return cv2.VideoCapture(camera_id)


def read_camera_frame(camera_id: int | None = None):
    index = int(settings["camera_id"]) if camera_id is None else int(camera_id)
    cap = _open_camera_capture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class ContinuousFrameReader:
    """Keep one video/camera handle open during a WebSocket analysis session."""

    def __init__(self, source: str, sample_video: str | None = None):
        self.source = source
        self.sample_video = sample_video
        self.cap = None
        self.fps = 0.0
        self.last_position_frames = 0.0
        self._camera_reconnect_attempts = 0
        self._camera_max_reconnect = 3

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Open camera with resolution and buffer optimization."""
        cap = _open_camera_capture(int(settings["camera_id"]))
        if cap is None or not cap.isOpened():
            return None
        # Set resolution to analysis target to avoid processing oversized frames
        # (phone cameras often default to 1920x1080 or higher)
        target_w = int(settings.get("analysis_width", 640))
        target_h = int(settings.get("analysis_height", 360))
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            # Minimize buffer for lowest latency (not all backends support this)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def __enter__(self):
        if self.source == "camera":
            self.cap = self._open_camera()
            self._camera_reconnect_attempts = 0
        else:
            video_path = resolve_student_video(self.sample_video)
            if video_path and video_path.exists():
                self.cap = cv2.VideoCapture(str(video_path))
        if self.cap is not None:
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def read(self, interval_ms: int = 0):
        if self.cap is None or not self.cap.isOpened():
            # Auto-reconnect for camera source
            if self.source == "camera" and self._camera_reconnect_attempts < self._camera_max_reconnect:
                self._camera_reconnect_attempts += 1
                self.cap = self._open_camera()
                if self.cap is None:
                    return None
            else:
                return None
        if self.source == "camera":
            latest = None
            for _ in range(3):
                ret, frame = self.cap.read()
                if ret:
                    latest = frame
                    self.last_position_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if latest is None and self._camera_reconnect_attempts < self._camera_max_reconnect:
                # Camera read failed, try reconnect
                self._camera_reconnect_attempts += 1
                self.cap.release()
                self.cap = self._open_camera()
                if self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        latest = frame
                        self._camera_reconnect_attempts = 0
            return latest
        if self.source == "sample" and interval_ms > 0 and self.fps > 1:
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            step = max(1, int(round(self.fps * interval_ms / 1000.0)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current + step - 1)
        ret, frame = self.cap.read()
        if ret:
            self.last_position_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return frame
        if self.source == "sample":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.last_position_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return frame if ret else None
        return None

    def position_text(self) -> str:
        if self.source == "sample" and self.fps > 0:
            seconds = self.last_position_frames / self.fps
            return f"video {seconds:05.1f}s / frame {int(self.last_position_frames)}"
        return f"camera frame {int(self.last_position_frames)}"


def resize_for_analysis(frame):
    max_width = int(settings.get("analysis_width", 640) or 640)
    max_height = int(settings.get("analysis_height", 360) or 360)
    height, width = frame.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def decode_browser_camera_frame(frame_bytes: bytes):
    """Decode one browser-uploaded JPEG frame and constrain its analysis size."""
    if not frame_bytes:
        raise ValueError("浏览器摄像头上传了空帧")
    if len(frame_bytes) > MAX_BROWSER_CAMERA_FRAME_BYTES:
        raise ValueError(
            f"浏览器摄像头帧超过 {MAX_BROWSER_CAMERA_FRAME_BYTES // 1024} KiB 限制"
        )
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("无法解码浏览器摄像头 JPEG 帧")
    return resize_for_analysis(frame)


def frame_to_data_url(frame) -> str:
    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def frame_to_thumbnail_data_url(frame, max_width: int = 320, quality: int = 65) -> str:
    """生成小尺寸缩略图 data URL，用于课堂管理端上报（减小带宽）。"""
    if frame is None:
        return ""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (max_width, max(1, int(h * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# 线下课堂管理端上报
# ---------------------------------------------------------------------------

def _store_classroom_report(
    payload: dict,
    annotated_frame,
    device_id: str,
    seat_label: str | None = None,
    student_name: str | None = None,
) -> None:
    """分析完一帧后，提取关键字段存入全局变量，供后台线程上报。"""
    report = payload.get("report", {})
    edge = payload.get("edge", {})
    dao = edge.get("dao", {})
    state_counts = report.get("state_counts", {})
    dominant = _dominant_state(state_counts)
    thumbnail = frame_to_thumbnail_data_url(annotated_frame)
    report_data = {
        "device_id": device_id or normalize_client_id(None),
        "seat_label": seat_label or settings.get("seat_label") or platform.node() or "未命名座位",
        "student_name": student_name or settings.get("student_name") or "未命名学生",
        "learning_state": dominant,
        "state_label": STATE_LABELS.get(dominant, dominant),
        "state_counts": state_counts,
        "needs_attention": report.get("needs_attention", []),
        "frame_thumbnail": thumbnail,
        "latency_ms": payload.get("elapsed_ms", 0.0),
        "dao_strategy": edge.get("strategy") or dao.get("strategy") or "unknown",
        "dao_admission": dao.get("admission"),
        "edge_connected": edge.get("connected", False),
        "timestamp": time.time(),
    }
    with _classroom_report_lock:
        _classroom_latest_reports[report_data["device_id"]] = report_data


def _remove_classroom_report(device_id: str) -> None:
    with _classroom_report_lock:
        _classroom_latest_reports.pop(device_id, None)


def _classroom_report_loop() -> None:
    """后台线程：每 3 秒将最新分析结果 POST 到课堂管理端服务器。"""
    while True:
        time.sleep(3)
        url = settings.get("classroom_dashboard_url", "").strip()
        if not url:
            continue
        with _classroom_report_lock:
            reports = list(_classroom_latest_reports.values())
        if not reports:
            continue
        for report in reports:
            try:
                full_url = url.rstrip("/") + "/api/student/report"
                data = json.dumps(report, ensure_ascii=False).encode("utf-8")
                req = urlrequest.Request(
                    full_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urlrequest.urlopen(req, timeout=5)
            except Exception:
                pass  # 管理端服务器可能尚未启动，静默忽略


def draw_monitor_overlay(frame, frame_index: int | None, position_text: str | None):
    if frame_index is None and not position_text:
        return frame
    label = f"EDGE SAMPLE #{frame_index or 0}"
    if position_text:
        label = f"{label} | {position_text}"
    h, w = frame.shape[:2]
    # 字号随帧宽自适应，避免固定 500px 矩形在小帧上溢出、在大帧上过小
    scale = max(0.4, min(0.85, w / 1000.0))
    thick = max(1, int(round(scale * 2.5)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    # 文字超出帧宽时截断（保留省略号），保证不越过视频右边界
    if tw > w - 24:
        cut = max(1, int(len(label) * (w - 24) / max(tw, 1)) - 1)
        label = label[:cut] + ".."
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    bar_h = th + 16
    rect_x2 = min(w - 10, 10 + tw + 20)
    cv2.rectangle(frame, (10, 10), (rect_x2, 10 + bar_h), (15, 23, 42), -1)
    cv2.putText(
        frame,
        label,
        (20, 10 + bar_h - th // 2 - 2),
        font,
        scale,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )
    return frame


def placeholder_frame(message: str):
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:] = (24, 32, 46)
    cv2.rectangle(frame, (40, 40), (920, 500), (72, 91, 117), 2)
    cv2.putText(frame, "Waiting for student video or camera", (90, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 238, 248), 2)
    cv2.putText(frame, message[:48], (90, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (150, 170, 196), 2)
    return frame


def placeholder_analysis(lesson_id: str, message: str) -> dict:
    lesson = find_lesson(lesson_id)
    report = {
        "total": 0,
        "focused": [],
        "needs_attention": [],
        "state_counts": {**{state: 0 for state in STATE_ORDER}, "absent": 1},
        "state_labels": STATE_LABELS,
        "strategy": "waiting",
        "total_ms": 0.0,
        "detections": 0,
    }
    return {
        "timestamp": time.time(),
        "elapsed_ms": 0.0,
        "report": report,
        "action": build_teaching_action(report, lesson),
        "image": frame_to_data_url(placeholder_frame(message)),
        "edge": current_edge_payload("waiting"),
    }


def sampling_placeholder_analysis(
    lesson_id: str,
    frame,
    frame_index: int,
    position_text: str | None,
    analysis_mode: str = "standard",
    playback_time: float | None = None,
    device_id: str | None = None,
    source: str | None = None,
) -> dict:
    mode = analysis_mode if analysis_mode in ANALYSIS_MODE_CONFIG else "standard"
    base_lesson = find_lesson(lesson_id)
    segment = resolve_lesson_segment(lesson_id, playback_time)
    preview = resize_for_analysis(frame)
    preview = draw_monitor_overlay(preview.copy(), frame_index, position_text)
    report = {
        "total": 0,
        "focused": [],
        "needs_attention": [],
        "state_counts": {state: 0 for state in STATE_ORDER},
        "state_labels": STATE_LABELS,
        "strategy": "sampling",
        "total_ms": 0.0,
        "detections": 0,
        "analysis_mode": mode,
        "pose_interval": ANALYSIS_MODE_CONFIG[mode]["pose_interval"],
    }
    return {
        "timestamp": time.time(),
        "elapsed_ms": 0.0,
        "sample": {
            "frame_index": frame_index,
            "position": position_text,
            "playback_time": playback_time,
            "analysis_mode": mode,
            "mode_label": ANALYSIS_MODE_CONFIG[mode]["label"],
            "pose_interval": ANALYSIS_MODE_CONFIG[mode]["pose_interval"],
            "source": source,
            "device_id": device_id,
        },
        "report": report,
        "image": frame_to_data_url(preview),
        "lesson_context": {
            "lesson_id": lesson_id,
            "title": base_lesson["title"],
            "playback_time": playback_time,
            "segment": segment,
            "knowledge": segment["topic"] if segment else base_lesson["knowledge"],
        },
        "edge": current_edge_payload("sampling"),
    }


def remember_non_absent_report(report: dict) -> dict:
    global last_non_absent_report, consecutive_empty_frames
    consecutive_empty_frames = 0
    last_non_absent_report = report
    return report


def stabilize_absent_report(report: dict) -> dict:
    global consecutive_empty_frames
    consecutive_empty_frames += 1
    threshold = int(os.getenv("ABSENT_CONFIRM_FRAMES", "4"))
    if consecutive_empty_frames < threshold and last_non_absent_report:
        return {
            **last_non_absent_report,
            "strategy": report.get("strategy", last_non_absent_report.get("strategy", "unknown")),
            "total_ms": report.get("total_ms", last_non_absent_report.get("total_ms", 0.0)),
            "detections": 0,
            "presence_status": f"检测短暂丢失 {consecutive_empty_frames}/{threshold}",
        }
    report["state_counts"]["absent"] = 1
    report["presence_status"] = f"连续 {consecutive_empty_frames} 帧未检测到学生"
    return report


def student_report(result: dict) -> dict:
    analysis = result.get("analysis_result")
    analysis_summary = result.get("analysis_summary")
    detections = result.get("detections", [])
    report = {
        "total": 0,
        "focused": [],
        "needs_attention": [],
        "state_counts": {state: 0 for state in STATE_ORDER},
        "state_labels": STATE_LABELS,
        "strategy": result.get("strategy", "unknown"),
        "total_ms": float(result.get("total_ms", 0.0) or 0.0),
        "detections": len(detections),
    }
    if isinstance(analysis_summary, dict):
        students = analysis_summary.get("students")
        if isinstance(students, list):
            for index, student in enumerate(students):
                learning_state = student.get("learning_state") or {}
                state = normalize_learning_state(
                    learning_state.get("label"),
                    has_detection=True,
                )
                item = {
                    "id": student.get("track_id", index + 1),
                    "state": state,
                    "state_label": STATE_LABELS.get(state, state),
                    "intervention": learning_state.get("intervention", "none"),
                    "confidence": learning_state.get("confidence", 0.0),
                    "behavior": (student.get("behavior") or {}).get("label", "unknown"),
                    "reasons": learning_state.get("reasons", []),
                    "offloaded": True,
                }
                report["total"] += 1
                if state in report["state_counts"]:
                    report["state_counts"][state] += 1
                if state == "focused":
                    report["focused"].append(item)
                else:
                    report["needs_attention"].append(item)
            if report["total"] == 0:
                return stabilize_absent_report(report)
            return remember_non_absent_report(report)

        learning_state = analysis_summary.get("learning_state")
        if isinstance(learning_state, dict):
            has_detection = bool(detections or analysis_summary.get("num_people_detected", 0))
            state = normalize_learning_state(learning_state.get("label"), has_detection=has_detection)
            if state == "absent":
                return stabilize_absent_report(report)
            else:
                report["total"] = max(1, int(analysis_summary.get("num_people_detected", 1) or 1))
                report["state_counts"][state] = report["total"]
                item = {
                    "id": 1,
                    "state": state,
                    "state_label": STATE_LABELS.get(state, state),
                    "intervention": learning_state.get("intervention", "none"),
                    "confidence": learning_state.get("confidence", 0.0),
                    "behavior": (analysis_summary.get("behavior") or {}).get("label", "unknown"),
                    "reasons": learning_state.get("reasons", []),
                    "offloaded": True,
                }
                if state == "focused":
                    report["focused"].append(item)
                else:
                    report["needs_attention"].append(item)
            return remember_non_absent_report(report)

    if not analysis or not hasattr(analysis, "students"):
        if detections:
            report["total"] = len(detections)
            report["state_counts"]["focused"] = len(detections)
            for index, _det in enumerate(detections):
                report["focused"].append({
                    "id": index + 1,
                    "state": "focused",
                    "state_label": STATE_LABELS["focused"],
                    "intervention": "none",
                })
            return remember_non_absent_report(report)
        else:
            return stabilize_absent_report(report)

    for index, student in enumerate(analysis.students):
        state = "unknown"
        intervention = "none"
        if hasattr(student, "learning_state"):
            state = getattr(student.learning_state, "label", "unknown")
            intervention = getattr(student.learning_state, "intervention", "none")
        state = normalize_learning_state(state, has_detection=True)
        item = {
            "id": getattr(student, "track_id", index + 1),
            "state": state,
            "state_label": STATE_LABELS.get(state, state),
            "intervention": intervention,
        }
        report["total"] += 1
        if state in report["state_counts"]:
            report["state_counts"][state] += 1
        if state == "focused":
            report["focused"].append(item)
        else:
            report["needs_attention"].append(item)
    if report["total"] == 0:
        return stabilize_absent_report(report)
    return remember_non_absent_report(report)


def choose_teaching_action(report: dict, lesson: dict) -> dict:
    states = [item["state"] for item in report["needs_attention"]]
    if report["total"] == 0 and report["detections"] == 0:
        return {
            "state": "缺勤",
            "action": "缺勤：暂停课程",
            "pace": "等待",
            "reason": "未检测到学生，无法确认学习状态。",
            "script": "我先暂停课程。请回到摄像头前，准备好后我们继续。",
            "next_step": "保持当前课程进度，等待学生回到学习画面。",
        }
    if "absent" in states:
        return {
            "state": "缺勤",
            "action": "缺勤：暂停课程并记录",
            "pace": "暂停",
            "reason": "检测到学生离开座位或不在镜头前。",
            "script": "我检测到你暂时离开了学习位置。课程将暂停，回到座位后我们从当前知识点继续。",
            "next_step": "暂停播放，保留当前进度。",
        }
    if "confused" in states:
        return {
            "state": "困惑",
            "action": "困惑：放慢节奏并重讲",
            "pace": "0.75x",
            "reason": f"学生可能对“{lesson['knowledge']}”存在理解困难。",
            "script": f"这里可能有些难。我会放慢速度，重新解释“{lesson['knowledge']}”，然后给你一个小问题确认理解。",
            "next_step": f"重讲当前知识点，并推送练习：{lesson['quiz']}",
        }
    if "tired" in states:
        return {
            "state": "疲劳",
            "action": "疲劳：提示可以考虑休息",
            "pace": lesson["pace"],
            "reason": "检测到疲劳迹象，仅给出休息提醒，不改变课程播放。",
            "script": "你看起来有些疲劳。如果感觉注意力下降，可以考虑暂停休息一下，调整坐姿和视线后再继续。",
            "next_step": "继续播放课程，必要时由学生主动暂停休息。",
        }
    if "distracted" in states:
        return {
            "state": "分心",
            "action": "分心：暂停课程并语音提醒",
            "pace": "暂停",
            "reason": "检测到注意力偏离，需要先暂停课程并用语音提醒学生回到学习状态。",
            "script": f"我先暂停课程，并播放一条提醒语音。请回到屏幕前后回答这个小问题：{lesson['quiz']}",
            "next_step": "暂停课程视频，前端切换为红色警示主题，并随机播放一条注意力提醒录音。",
        }
    return {
        "state": "专注",
        "action": "专注：继续教学",
        "pace": lesson["pace"],
        "reason": "学生保持专注，当前节奏合适。",
        "script": "你保持得很好，我们继续学习下一小节。",
        "next_step": "继续播放教学内容，必要时进入巩固练习。",
    }


def llm_feedback_enabled() -> bool:
    return bool(settings["llm_enabled"] and settings.get("llm_api_key"))


def feedback_cache_key(action: dict, report: dict, lesson: dict) -> str:
    states = [item["state"] for item in report.get("needs_attention", [])]
    payload = {
        "lesson_id": lesson["id"],
        "action": action.get("action"),
        "pace": action.get("pace"),
        "states": sorted(states),
        "total": report.get("total", 0),
        "detections": report.get("detections", 0),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def call_teacher_llm(action: dict, report: dict, lesson: dict) -> dict:
    api_key = settings.get("llm_api_key", "")
    if not llm_feedback_enabled():
        return {**action, "feedback_source": "规则引擎", "llm_used": False}

    key = feedback_cache_key(action, report, lesson)
    cached = feedback_cache.get(key)
    if cached and time.time() - cached["created_at"] < 300:
        return cached["action"]

    state_summary = {
        "student_count": report.get("total") or report.get("detections") or 0,
        "learning_state_counts": {
            STATE_LABELS[state]: int(report.get("state_counts", {}).get(state, 0))
            for state in STATE_ORDER
        },
        "focused_count": len(report.get("focused", [])),
        "needs_attention": [
            {
                "id": item.get("id"),
                "state": STATE_LABELS.get(item.get("state"), item.get("state")),
                "intervention": item.get("intervention"),
            }
            for item in report.get("needs_attention", [])
        ],
        "latency_ms": round(float(report.get("total_ms", 0.0) or 0.0), 1),
    }
    system_prompt = (
        "你是一个面向大学高等数学网课的 AI 虚拟教师。"
        "你只能生成课堂反馈话术和下一步教学安排，不能改变系统已经给出的教学动作、播放节奏或 DAO 卸载决策。"
        "输出必须是严格 JSON，不要 Markdown，不要解释。"
    )
    user_prompt = {
        "课程": {
            "标题": lesson["title"],
            "知识点": lesson["knowledge"],
            "当前片段": lesson.get("current_segment") or {},
            "讲解要点": lesson["content"],
            "当前练习": lesson["quiz"],
        },
        "学生状态": state_summary,
        "系统规则决策": {
            "action": action["action"],
            "pace": action["pace"],
            "reason": action["reason"],
            "script": action["script"],
            "next_step": action["next_step"],
        },
        "生成要求": {
            "reason": "用一句话解释为什么这样干预，控制在 35 个汉字以内。",
            "script": "生成自然、鼓励式、适合网课场景的教师话术，控制在 80 个汉字以内。",
            "next_step": "生成具体下一步教学安排，控制在 80 个汉字以内。",
            "必须保持": ["action", "pace"],
        },
        "返回 JSON 格式": {
            "reason": "string",
            "script": "string",
            "next_step": "string",
        },
    }

    body = {
        "model": settings["llm_model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.4,
        "max_tokens": 260,
        "response_format": {"type": "json_object"},
    }
    endpoint = f"{settings['llm_api_base']}/chat/completions"
    request = urlrequest.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=float(settings["llm_timeout_seconds"])) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        generated = parse_llm_json(content)
        enhanced = {
            **action,
            "reason": str(generated.get("reason") or action["reason"]),
            "script": str(generated.get("script") or action["script"]),
            "next_step": str(generated.get("next_step") or action["next_step"]),
            "feedback_source": f"大模型：{settings['llm_model']}",
            "llm_used": True,
        }
        if len(feedback_cache) > 128:
            feedback_cache.clear()
        feedback_cache[key] = {"created_at": time.time(), "action": enhanced}
        return enhanced
    except (urlerror.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {
            **action,
            "feedback_source": "规则引擎",
            "llm_used": False,
            "llm_error": str(exc),
        }


def teaching_state_signature(action: dict, report: dict, lesson: dict) -> str:
    needs_attention = [
        {
            "id": item.get("id"),
            "state": item.get("state"),
            "intervention": item.get("intervention"),
        }
        for item in report.get("needs_attention", [])
    ]
    focused = [item.get("id") for item in report.get("focused", [])]
    return json.dumps(
        {
            "lesson_id": lesson.get("id"),
            "segment": lesson.get("current_segment", {}).get("topic"),
            "action": action.get("action"),
            "pace": action.get("pace"),
            "student_count": report.get("total") or report.get("detections") or 0,
            "focused": focused,
            "needs_attention": needs_attention,
            "strategy": report.get("strategy"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def build_teaching_action(report: dict, lesson: dict) -> dict:
    base_action = choose_teaching_action(report, lesson)
    signature = teaching_state_signature(base_action, report, lesson)
    cached_signature = last_teaching_feedback.get("signature")
    cached_action = last_teaching_feedback.get("action")
    if cached_signature == signature and isinstance(cached_action, dict):
        return {
            **cached_action,
            "feedback_source": cached_action.get("feedback_source", "规则引擎"),
            "llm_reused": bool(cached_action.get("llm_used")),
        }

    action = call_teacher_llm(base_action, report, lesson)
    last_teaching_feedback["signature"] = signature
    last_teaching_feedback["action"] = action
    return action


def analyze_frame(
    frame,
    lesson_id: str,
    frame_index: int | None = None,
    position_text: str | None = None,
    analysis_mode: str = "standard",
    playback_time: float | None = None,
    device_id: str | None = None,
    source: str | None = None,
) -> dict:
    mode = analysis_mode if analysis_mode in ANALYSIS_MODE_CONFIG else "standard"
    base_lesson = find_lesson(lesson_id)
    segment = resolve_lesson_segment(lesson_id, playback_time)
    lesson = lesson_with_segment(base_lesson, segment)
    edu = get_analyzer()
    start = time.time()
    analysis_frame = resize_for_analysis(frame)
    result = edu.analyze_frame(analysis_frame, analysis_mode=mode, device_id=device_id)
    annotated = edu.draw_results(analysis_frame.copy(), result)
    annotated = draw_monitor_overlay(annotated, frame_index, position_text)
    report = student_report(result)
    report["analysis_mode"] = result.get("analysis_mode", mode)
    report["pose_interval"] = result.get("pose_interval", ANALYSIS_MODE_CONFIG[mode]["pose_interval"])
    report["pose_refreshed"] = bool(result.get("pose_refreshed", False))
    action = build_teaching_action(report, lesson)
    payload = {
        "timestamp": time.time(),
        "elapsed_ms": round((time.time() - start) * 1000, 1),
        "sample": {
            "frame_index": frame_index,
            "position": position_text,
            "playback_time": playback_time,
            "analysis_mode": mode,
            "mode_label": ANALYSIS_MODE_CONFIG[mode]["label"],
            "pose_interval": result.get("pose_interval", ANALYSIS_MODE_CONFIG[mode]["pose_interval"]),
            "pose_refreshed": bool(result.get("pose_refreshed", False)),
            "source": source,
        },
        "report": report,
        "action": action,
        "lesson_context": {
            "lesson_id": lesson_id,
            "title": base_lesson["title"],
            "playback_time": playback_time,
            "segment": segment,
            "knowledge": lesson["knowledge"],
            "quiz": lesson["quiz"],
        },
        "image": frame_to_data_url(annotated),
        "edge": {
            **current_edge_payload(getattr(edu, "strategy", "unknown")),
            "dao": result.get("dao_decision") or getattr(edu, "dao_decision_info", {}),
        },
    }
    record_scheduler_event(payload, device_id=device_id, source=source)
    _store_classroom_report(payload, annotated, device_id or normalize_client_id(None))
    return payload


# ---------------------------------------------------------------------------
# 线下课堂专用：轻量分析函数（不依赖课程目录 / 网课播放）
# ---------------------------------------------------------------------------

def analyze_classroom_frame(
    frame,
    frame_index: int | None = None,
    position_text: str | None = None,
    analysis_mode: str = "standard",
    device_id: str | None = None,
    source: str | None = None,
    seat_label: str | None = None,
    student_name: str | None = None,
) -> dict:
    """线下课堂专用分析函数。

    与 ``analyze_frame`` 相比，跳过所有课程相关逻辑（课程目录查找、
    播放时间解析、教学动作生成），只做：YOLO 检测 → 学习状态分析 →
    DAO 卸载 → 生成缩略图 → 触发管理端上报。
    """
    mode = analysis_mode if analysis_mode in ANALYSIS_MODE_CONFIG else "standard"
    edu = get_analyzer()
    start = time.time()
    # 分析用缩放帧（低分辨率，加速检测）；显示用原始帧（高分辨率，更清晰）
    analysis_frame = resize_for_analysis(frame)
    result = edu.analyze_frame(analysis_frame, analysis_mode=mode, device_id=device_id)
    h0, w0 = frame.shape[:2]
    h1, w1 = analysis_frame.shape[:2]
    if w1 != w0 or h1 != h0:
        # 把检测/跟踪坐标按缩放比例映射回原始分辨率，保证标注框位置正确
        sx = w0 / max(w1, 1)
        sy = h0 / max(h1, 1)
        display_result = dict(result)
        display_result["detections"] = [
            {**d, "x1": d["x1"] * sx, "y1": d["y1"] * sy, "x2": d["x2"] * sx, "y2": d["y2"] * sy}
            for d in result.get("detections", [])
        ]
        display_result["tracks"] = [
            {**t, "x1": t["x1"] * sx, "y1": t["y1"] * sy, "x2": t["x2"] * sx, "y2": t["y2"] * sy}
            for t in result.get("tracks", [])
        ]
        annotated = edu.draw_results(frame.copy(), display_result)
    else:
        annotated = edu.draw_results(frame.copy(), result)
    annotated = draw_monitor_overlay(annotated, frame_index, position_text)
    report = student_report(result)
    report["analysis_mode"] = result.get("analysis_mode", mode)
    report["pose_interval"] = result.get("pose_interval", ANALYSIS_MODE_CONFIG[mode]["pose_interval"])
    report["pose_refreshed"] = bool(result.get("pose_refreshed", False))
    payload = {
        "timestamp": time.time(),
        "elapsed_ms": round((time.time() - start) * 1000, 1),
        "sample": {
            "frame_index": frame_index,
            "position": position_text,
            "analysis_mode": mode,
            "mode_label": ANALYSIS_MODE_CONFIG[mode]["label"],
            "pose_interval": result.get("pose_interval", ANALYSIS_MODE_CONFIG[mode]["pose_interval"]),
            "pose_refreshed": bool(result.get("pose_refreshed", False)),
            "source": source,
        },
        "report": report,
        "image": frame_to_data_url(annotated),
        "edge": {
            **current_edge_payload(getattr(edu, "strategy", "unknown")),
            "dao": result.get("dao_decision") or getattr(edu, "dao_decision_info", {}),
        },
    }
    record_scheduler_event(payload, device_id=device_id, source=source)
    _store_classroom_report(
        payload,
        annotated,
        device_id or normalize_client_id(None),
        seat_label=seat_label,
        student_name=student_name,
    )
    return payload


@app.get("/client")
def classroom_client_page():
    """线下课堂学生端轻量页面。"""
    index = CLASSROOM_CLIENT_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "前端文件未找到"}, status_code=404)


@app.websocket("/ws/classroom-stream")
async def classroom_stream_ws(websocket: WebSocket):
    """线下课堂专用 WebSocket 端点。

    与 ``/ws/analysis`` 相比，此端点**不需要 lesson_id**，不会播放任何网课视频，
    支持服务器本地摄像头、服务器样例视频，以及浏览器上传的本机摄像头帧。
    """
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        source = config.get("source", "camera")
        if source not in {"browser_camera", "camera", "sample"}:
            raise ValueError(f"不支持的课堂采集源: {source}")
        device_id = normalize_client_id(config.get("device_id"))
        seat_label = str(config.get("seat_label") or "").strip()
        student_name = str(config.get("student_name") or "").strip()
        sample_video = config.get("sample_video")
        analysis_mode = config.get("analysis_mode", "standard")
        if analysis_mode not in ANALYSIS_MODE_CONFIG:
            analysis_mode = "standard"
        interval_ms = max(300, int(config.get("interval_ms", 1500)))
        # max_frames=0 表示不限上限（持续分析），不是只分析 1 帧
        raw_max = int(config.get("max_frames", 0))
        max_frames = raw_max if raw_max > 0 else 100000

        await websocket.send_text(json.dumps({
            "type": "started",
            "source": source,
            "device_id": device_id,
            "analysis_mode": analysis_mode,
            "mode_label": ANALYSIS_MODE_CONFIG[analysis_mode]["label"],
            "pose_interval": ANALYSIS_MODE_CONFIG[analysis_mode]["pose_interval"],
        }, ensure_ascii=False))

        if source == "browser_camera":
            index = 0
            while True:
                if index >= max_frames:
                    break
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                text_message = message.get("text")
                if text_message is not None:
                    try:
                        update = json.loads(text_message)
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "frame_error",
                            "message": "浏览器摄像头控制消息不是有效 JSON",
                        }, ensure_ascii=False))
                        continue
                    if update.get("type") == "stop":
                        break
                    continue
                frame_bytes = message.get("bytes")
                if frame_bytes is None:
                    continue
                try:
                    frame = decode_browser_camera_frame(frame_bytes)
                except ValueError as exc:
                    await websocket.send_text(json.dumps({
                        "type": "frame_error",
                        "message": str(exc),
                    }, ensure_ascii=False))
                    continue

                index += 1
                frame_index = index
                payload = await asyncio.to_thread(
                    analyze_classroom_frame,
                    frame,
                    frame_index,
                    f"browser camera frame {frame_index}",
                    analysis_mode,
                    device_id,
                    source,
                    seat_label,
                    student_name,
                )
                payload["type"] = "analysis"
                payload["frame_index"] = frame_index
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        else:
            with ContinuousFrameReader(source, sample_video) as reader:
                index = 0
                while True:
                    index += 1
                    if index > max_frames:
                        break
                    try:
                        update = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                        if update.get("type") == "stop":
                            break
                    except asyncio.TimeoutError:
                        pass

                    frame_index = index
                    frame = reader.read(interval_ms)
                    if frame is None:
                        payload = {
                            "type": "analysis",
                            "frame_index": frame_index,
                            "report": {"state_counts": {"absent": 1}, "state_labels": STATE_LABELS, "detections": 0},
                            "image": "",
                            "edge": current_edge_payload("no_source"),
                        }
                        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                        await asyncio.sleep(interval_ms / 1000)
                        continue

                    position_text = reader.position_text()
                    payload = await asyncio.to_thread(
                        analyze_classroom_frame,
                        frame,
                        frame_index,
                        position_text,
                        analysis_mode,
                        device_id,
                        source,
                        seat_label,
                        student_name,
                    )
                    payload["type"] = "analysis"
                    payload["frame_index"] = frame_index
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                    await asyncio.sleep(interval_ms / 1000)

        await websocket.send_text(json.dumps({"type": "finished"}, ensure_ascii=False))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
    finally:
        if "device_id" in locals():
            _remove_classroom_report(device_id)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/admin")
def admin_index():
    return FileResponse(ADMIN_FRONTEND_DIR / "index.html")


@app.get("/api/lessons")
def lessons():
    return {"lessons": [lesson_payload(lesson) for lesson in LESSON_CATALOG]}


@app.get("/api/lesson-video/{lesson_id}")
def lesson_video(lesson_id: str):
    lesson = find_lesson(lesson_id)
    video_path = LESSON_DIR / lesson["local_video"]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="教学视频不存在")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/student-videos")
def student_videos():
    return {"videos": list_student_videos()}


@app.get("/api/cameras")
def list_cameras():
    """Detect available camera devices on this machine."""
    cameras = []
    for index in range(8):  # 检查索引 0-7
        # 注意：枚举探测必须用默认后端（MSMF）。DSHOW 打开不存在的索引会触发
        # OpenCV 原生崩溃（整个服务进程挂掉），仅用于打开已确认存在的 camera_id。
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            info = {"index": index, "opened": True}
            if ret and frame is not None:
                h, w = frame.shape[:2]
                info["width"] = int(w)
                info["height"] = int(h)
            cameras.append(info)
        if cap is not None:
            cap.release()
    return {
        "cameras": cameras,
        "current_camera_id": int(settings["camera_id"]),
    }


@app.get("/api/camera/preview")
def camera_preview(index: int | None = None):
    """Capture one frame from the current camera and return as JPEG data URL.

    可通过 ?index=N 指定摄像头索引做快速自检（例如 /api/camera/preview?index=1）。
    """
    frame = read_camera_frame(index)
    if frame is None:
        return {"ok": False, "error": "无法打开摄像头，请检查 camera_id 设置或设备连接"}
    frame = resize_for_analysis(frame)
    return {"ok": True, "image": frame_to_data_url(frame)}


@app.get("/api/status")
def status():
    return {
        "runtime_os": platform.system(),
        "runtime_host": platform.node(),
        "service_role": "education-web",
        # 学生端（/client）用该 ID 向管理端查询个人课堂学习报告，
        # 与 _classroom_report_loop 上报的 device_id 保持同一口径
        "device_id": normalize_client_id(None),
        "settings": public_settings(),
        "edge": current_edge_payload(),
        "llm": {
            "enabled": bool(settings["llm_enabled"]),
            "configured": bool(settings.get("llm_api_key")),
            "model": settings["llm_model"],
            "api_base": settings["llm_api_base"],
            "timeout_seconds": settings["llm_timeout_seconds"],
        },
        "device": {
            "cpu": DeviceProfiler.get_cpu_percent(interval=0.1),
            "memory": DeviceProfiler.get_memory_percent(),
            "score": DeviceProfiler.estimate_performance_score(),
        },
    }


@app.get("/api/admin/snapshot")
def admin_snapshot():
    return admin_snapshot_payload()


@app.get("/api/admin/dao-snapshot")
def admin_dao_snapshot():
    """调度页数据：优先返回全课堂真实设备，教师端不可达时退回本机。"""
    return classroom_admin_snapshot() or admin_snapshot_payload()


@app.put("/api/settings")
def update_settings(payload: SettingsRequest):
    if payload.edge_host is not None:
        settings["edge_host"] = payload.edge_host.strip()
    if payload.edge_port is not None:
        settings["edge_port"] = int(payload.edge_port)
    if payload.use_edge is not None:
        settings["use_edge"] = bool(payload.use_edge)
    if payload.camera_id is not None:
        settings["camera_id"] = int(payload.camera_id)
    if payload.student_video_dir:
        settings["student_video_dir"] = payload.student_video_dir
    if payload.llm_timeout_seconds is not None:
        settings["llm_timeout_seconds"] = max(5.0, min(60.0, float(payload.llm_timeout_seconds)))
    if payload.classroom_dashboard_url:
        settings["classroom_dashboard_url"] = payload.classroom_dashboard_url
    if payload.seat_label:
        settings["seat_label"] = payload.seat_label
    if payload.student_name:
        settings["student_name"] = payload.student_name
    reset_analyzer()
    return {"ok": True, "settings": public_settings()}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    frame = read_camera_frame() if payload.source == "camera" else read_sample_frame(payload.sample_video)
    if frame is None:
        return placeholder_analysis(LESSON_CATALOG[0]["id"], "No student source available")
    return analyze_frame(
        frame,
        LESSON_CATALOG[0]["id"],
        device_id=normalize_client_id(payload.device_id),
        source=payload.source,
    )


CHAT_API_KEY = os.getenv("LLM_API_KEY") or settings.get("llm_api_key", "")


def build_chat_fallback(message: str, lesson: dict | None, reason: str) -> dict:
    lesson_hint = ""
    if lesson:
        lesson_hint = f"当前课程是《{lesson.get('title', '')}》，核心知识点是：{lesson.get('knowledge', '')}。"
    response = (
        f"AI 接口暂时连接较慢（{reason}）。"
        f"{lesson_hint}"
        "我先用本地规则给出建议：请先继续当前学习流程；如果是概念题，可以把不理解的定义、例题步骤或时间点发给我，我会在网络恢复后继续帮你细化讲解。"
    )
    return {
        "response": response,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "fallback": True,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest):
    """AI 聊天端点：接收用户消息，调用 OpenAI/DeepSeek 兼容接口，返回回复与 token 消耗。"""
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    lesson = None
    if payload.lesson_id:
        lesson = next(
            (l for l in LESSON_CATALOG if l["id"] == payload.lesson_id), None
        )

    if not CHAT_API_KEY:
        return build_chat_fallback(payload.message, lesson, "未配置 API Key")

    system_prompt = (
        "你是一个面向大学高等数学网课的 AI 虚拟教师助手。"
        "你可以回答学生关于课程内容、数学概念、学习方法等方面的问题。"
        "回答应该简洁明了、鼓励式的，适合网课场景。"
        "如果用户的问题涉及当前课程内容，请结合课程信息回答。"
    )
    if lesson:
        system_prompt += (
            f"\n\n当前课程信息：\n"
            f"- 标题：{lesson.get('title', '')}\n"
            f"- 知识点：{lesson.get('knowledge', '')}\n"
            f"- 内容：{lesson.get('content', '')}\n"
            f"- 练习：{lesson.get('quiz', '')}"
        )

    messages = [{"role": "system", "content": system_prompt}]
    if payload.history:
        messages.extend(payload.history)
    messages.append({"role": "user", "content": payload.message})

    body = {
        "model": payload.model,
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 360,
    }

    endpoint = f"{payload.api_base.rstrip('/')}/chat/completions"
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {CHAT_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        timeout_seconds = max(5.0, min(15.0, float(settings.get("llm_timeout_seconds", 15))))
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return {
            "response": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
    except urlerror.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise HTTPException(
            status_code=exc.code,
            detail=f"AI API 调用失败 ({exc.code}): {err_body[:300]}",
        )
    except TimeoutError as exc:
        return build_chat_fallback(payload.message, lesson, f"读取超时：{exc}")
    except urlerror.URLError as exc:
        return build_chat_fallback(payload.message, lesson, f"网络不可用：{exc}")
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"AI API 返回格式异常: {exc}")


@app.websocket("/ws/analysis")
async def analysis_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        source = config.get("source", "sample")
        device_id = normalize_client_id(config.get("device_id"))
        lesson_id = config.get("lesson_id", LESSON_CATALOG[0]["id"])
        sample_video = config.get("sample_video")
        analysis_mode = config.get("analysis_mode", "standard")
        if analysis_mode not in ANALYSIS_MODE_CONFIG:
            analysis_mode = "standard"
        playback_time = config.get("playback_time")
        interval_ms = max(300, int(config.get("interval_ms", 1200)))
        max_frames = max(1, min(int(config.get("max_frames", 120)), 1000))

        await websocket.send_text(json.dumps({
            "type": "started",
            "source": source,
            "analysis_mode": analysis_mode,
            "mode_label": ANALYSIS_MODE_CONFIG[analysis_mode]["label"],
            "pose_interval": ANALYSIS_MODE_CONFIG[analysis_mode]["pose_interval"],
        }, ensure_ascii=False))
        with ContinuousFrameReader(source, sample_video) as reader:
            for index in range(max_frames):
                try:
                    update = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                    if update.get("type") == "stop":
                        break
                    if update.get("type") == "playback":
                        playback_time = update.get("playback_time", playback_time)
                except asyncio.TimeoutError:
                    pass
                frame_index = index + 1
                frame = reader.read(interval_ms)
                if frame is None:
                    payload = placeholder_analysis(lesson_id, "No student source available")
                else:
                    position_text = reader.position_text()
                    preview = sampling_placeholder_analysis(
                        lesson_id,
                        frame,
                        frame_index,
                        position_text,
                        analysis_mode,
                        playback_time,
                        device_id,
                        source,
                    )
                    preview["type"] = "analysis"
                    preview["frame_index"] = frame_index
                    await websocket.send_text(json.dumps(preview, ensure_ascii=False))
                    try:
                        update = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                        if update.get("type") == "stop":
                            break
                        if update.get("type") == "playback":
                            playback_time = update.get("playback_time", playback_time)
                    except asyncio.TimeoutError:
                        pass
                    payload = await asyncio.to_thread(
                        analyze_frame,
                        frame,
                        lesson_id,
                        frame_index,
                        position_text,
                        analysis_mode,
                        playback_time,
                        device_id,
                        source,
                    )
                payload["type"] = "analysis"
                payload["frame_index"] = frame_index
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                await asyncio.sleep(interval_ms / 1000)
        await websocket.send_text(json.dumps({"type": "finished"}, ensure_ascii=False))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
