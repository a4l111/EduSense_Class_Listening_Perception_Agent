#!/usr/bin/env python3
"""
线下课堂教师管理端服务（管理端，端口 8008）

接收各个树莓派（每台对应一个学生）上报的学习状态，
在统一的管理端页面上展示所有学生的实时学习情况。

端口分工
--------
本项目采用**两端口分组**架构：
  - **管理端 :8008**（本文件）—— 教师总览 + 全课堂调度数据汇总
  - **教学端 :8009**（web_api.py）—— 网课模式（/）+ 学生采集端（/client）

本服务负责：
  1. 接收各树莓派 / 教学端 POST 上报的学习状态；
  2. 提供教师总览页面（/dashboard）与管控台壳页（/）；
  3. 维护学生卡片存储 + 状态告警逻辑。

启动方式（在 Windows 边缘服务器主机上运行）::

    python classroom_dashboard.py --port 8008

数据流::

    树莓派1 (学生A) ──HTTP POST──┐
    树莓派2 (学生B) ──HTTP POST──┼──> 管理端 (:8008) ──> 教师浏览器
    教学端 :8009    ──HTTP POST──┘    (3秒/次自动上报)

各树莓派端的 ``web_api.py`` 在分析完每一帧后会自动 POST 到此服务。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib import request as urlrequest

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CLASSROOM_WEB_DIR = BASE_DIR / "web" / "classroom"
ADMIN_FRONTEND_DIR = BASE_DIR / "web" / "dao-admin"
VUE_FRONTEND_DIR = BASE_DIR / "web" / "vue-teacher"

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

# 需要关注的负面状态（会在告警面板高亮）
ATTENTION_STATES = {"distracted", "absent", "tired"}

# 学生上报超时（秒），超过此时间未上报则标记为"离线"
STUDENT_TIMEOUT_SEC = 30

# ---------------------------------------------------------------------------
# 全局状态（线程安全）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_students: OrderedDict[str, dict] = OrderedDict()
_alert_log: list[dict] = []
MAX_ALERT_LOG = 200

# ---------------------------------------------------------------------------
# 课堂历史持久化（SQLite）与挑战杯课堂中心页数据支持
# ---------------------------------------------------------------------------

HISTORY_DB_PATH = BASE_DIR / "data" / "education_history.db"
EDUCATION_GRPC_ADMIN_URL = os.getenv(
    "EDUCATION_GRPC_ADMIN_URL", "http://127.0.0.1:8011"
).rstrip("/")
SESSION_IDLE_SEC = 300.0

_history_lock = threading.Lock()
_session_lock = threading.Lock()
_session_state: dict = {"started_at": None, "last_report_at": None, "ended_at": None}

# 教师手动会话（开始上课 / 下课·生成报告）与学生时间线、最终报告、Ovis 复核。
# 内存态：服务重启后清空，历史明细仍由上方 SQLite 三表持久化。
# generating：下课后「正在生成报告」异步标记；error：报告生成失败原因（降级记录）。
_teacher_session: dict = {
    "active": False,
    "started_at": None,
    "ended_at": None,
    "report": None,
    "generating": False,
    "error": None,
}
_student_timelines: dict[str, list[dict]] = {}
_student_reviews: dict[str, list[dict]] = {}
_classroom_reports: list[dict] = []
_student_final_reports: dict[str, dict] = {}
MAX_TIMELINE_EVENTS = 600
MAX_CLASSROOM_REPORTS = 50


def _history_conn() -> sqlite3.Connection:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _init_history_db() -> None:
    """初始化课堂历史库：事件流、报告流、调度任务流三张表。"""
    with _history_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS classroom_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                device_id TEXT NOT NULL,
                student_name TEXT,
                seat_label TEXT,
                learning_state TEXT,
                state_label TEXT,
                latency_ms REAL DEFAULT 0,
                dao_strategy TEXT,
                edge_connected INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_cls_events_ts ON classroom_events(timestamp DESC);

            CREATE TABLE IF NOT EXISTS classroom_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                device_id TEXT NOT NULL,
                student_name TEXT,
                seat_label TEXT,
                engine TEXT,
                report_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cls_reports_ts ON classroom_reports(timestamp DESC);

            CREATE TABLE IF NOT EXISTS classroom_tasks (
                sequence TEXT PRIMARY KEY,
                device_id TEXT,
                frame_index INTEGER,
                status TEXT,
                queue_wait_ms REAL DEFAULT 0,
                total_ms REAL DEFAULT 0,
                slo_met INTEGER,
                learning_state TEXT,
                arrived_at REAL,
                updated_at REAL
            );
            """
        )


def _record_event(entry: dict) -> None:
    try:
        with _history_lock, _history_conn() as conn:
            conn.execute(
                "INSERT INTO classroom_events"
                " (timestamp, device_id, student_name, seat_label, learning_state,"
                "  state_label, latency_ms, dao_strategy, edge_connected)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.get("timestamp"),
                    entry.get("device_id"),
                    entry.get("student_name"),
                    entry.get("seat_label"),
                    entry.get("learning_state"),
                    entry.get("state_label"),
                    entry.get("latency_ms") or 0,
                    entry.get("dao_strategy"),
                    1 if entry.get("edge_connected") else 0,
                ),
            )
    except Exception:
        pass


def _record_report(entry: dict) -> None:
    try:
        with _history_lock, _history_conn() as conn:
            conn.execute(
                "INSERT INTO classroom_reports"
                " (timestamp, device_id, student_name, seat_label, engine, report_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.get("timestamp"),
                    entry.get("device_id"),
                    entry.get("student_name"),
                    entry.get("seat_label"),
                    entry.get("engine") or "规则汇总",
                    json.dumps(entry, ensure_ascii=False),
                ),
            )
    except Exception:
        pass


def _fetch_admin_snapshot(timeout: float = 2.5) -> dict | None:
    """转发教育 gRPC 服务器(:8011)的 DAO 调度快照；不可达时返回 None。"""
    try:
        with urlrequest.urlopen(
            EDUCATION_GRPC_ADMIN_URL + "/api/snapshot", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _short_device_label(device_id: str) -> str:
    """把 client_rasp4-6__user-default 压缩成可读短名 rasp4-6。"""
    text = str(device_id or "unknown")
    text = text.split("__", 1)[0]
    return text[len("client_"):] if text.startswith("client_") else text


def _adapt_admin_snapshot(snap: dict) -> dict:
    """把 :8011 教育 DAO 快照适配成 dao-admin 前端（/admin 全局调度管理页）结构。

    前端期望（见 web/dao-admin/app.js fallback）：
      devices [{id,name,role,status,tasks,latency_ms,strategy,state}]
      events  [{id,time_label,device_id,strategy,admission,latency_ms,queue,
                timestamp(epoch 秒),frame_index,status,planned_schedule,
                actual_schedule,stage_timings}]
      summary/server/dao/pipeline/comparison 各字段与 fallback 同名。
    """
    summary = snap.get("summary") or {}
    resources = snap.get("resources") or {}
    server = snap.get("server") or {}
    dao = snap.get("dao") or {}
    now = time.time()
    avg_latency = float(summary.get("avg_latency_ms") or 0.0)
    # 管理页按挑战杯要求统一检查 200ms，不改变底层调度器自身的截止时间参数。
    slo_ms = 200.0

    with _lock:
        student_lookup = dict(_students)

    devices = []
    for item in snap.get("devices") or []:
        raw_id = str(item.get("device_id") or "")
        if not raw_id:
            continue
        short_id = _short_device_label(raw_id)
        age = float(item.get("last_seen_age_sec") or 0.0)
        if age <= 15.0:
            status = "active"
        elif age <= 300.0:
            status = "online"
        else:
            status = "standby"
        student = student_lookup.get(raw_id) or {}
        seat = student.get("seat_label") or ""
        state = student.get("state_label") or STATE_LABELS.get(
            str(item.get("last_state") or ""), ""
        )
        devices.append({
            "id": short_id,
            "name": f"用户 {short_id}",
            "role": "课堂采集端" + (f" · {seat}" if seat else ""),
            "status": status,
            "tasks": int(item.get("total_requests") or 0),
            "latency_ms": round(float(item.get("last_latency_ms") or 0.0), 1),
            "strategy": "offload",
            "state": state or "—",
        })

    events = []
    # :8011 使用 appendleft 保存任务，列表开头才是最新数据。此前错误截取末尾
    # 120 条，三张“近 20/60 秒”时间线拿到的全是旧任务，因而看起来完全不动。
    recent = list(snap.get("recent_requests") or [])[:120]
    for idx, item in enumerate(reversed(recent)):
        arrived = float(item.get("arrived_at") or 0.0)
        status = str(item.get("status") or "")
        if status == "completed":
            admission = "accept"
        elif status == "failed":
            admission = "reject"
        else:
            admission = ""
        events.append({
            "id": str(item.get("sequence") or item.get("request_id") or f"e{idx}"),
            "request_id": str(item.get("request_id") or ""),
            "sequence": item.get("sequence"),
            "time_label": (
                time.strftime("%H:%M:%S", time.localtime(arrived)) if arrived else "--:--:--"
            ),
            "device_id": _short_device_label(str(item.get("device_id") or "")),
            "strategy": "offload",
            "admission": admission,
            "latency_ms": round(float(item.get("total_ms") or 0.0), 1),
            "queue": int(item.get("queue_wait_ms") or 0),
            "timestamp": arrived or now,
            "frame_index": item.get("frame_index"),
            "status": status or "waiting",
            # 三张时间线图使用教育推理服务记录的真实阶段时间；字段原样保留，
            # 只在浏览器中把专业阶段名翻译成面向普通用户的中文。
            "planned_schedule": list(item.get("planned_schedule") or []),
            "actual_schedule": list(item.get("actual_schedule") or []),
            "stage_timings": dict(item.get("stage_timings") or {}),
        })

    priorities = [
        float(item.get("dao_priority"))
        for item in (snap.get("recent_requests") or [])
        if item.get("dao_priority") is not None
    ]
    if priorities:
        avg_priority = sum(priorities) / len(priorities)
        priority = avg_priority if avg_priority > 1.0 else avg_priority * 100.0
        priority = max(0.0, min(100.0, round(priority, 1)))
    else:
        priority = 0.0

    pipeline = [
        {"id": "sample", "name": "视频采样", "model": "读取课堂画面", "placement": "local", "offloaded": False},
        {"id": "detect", "name": "人物与物品检测", "model": "识别人、手机和书本", "placement": "edge", "offloaded": True},
        {"id": "pose", "name": "姿态与行为分析", "model": "识别姿态和课堂行为", "placement": "edge", "offloaded": True},
        {"id": "state", "name": "学习状态判断", "model": "综合表情、视线和行为", "placement": "edge", "offloaded": True},
        {"id": "teacher", "name": "AI 教师干预", "model": "规则/大模型反馈", "placement": "edge", "offloaded": True},
    ]

    baseline = round(max(avg_latency * 1.35, avg_latency + 80.0), 1)
    node = server.get("node_id") or "windows-edge-server"
    port = int(server.get("grpc_port") or 50051)

    return {
        "connected": True,
        "timestamp": now,
        "summary": {
            "total_requests": int(summary.get("total_requests") or 0),
            "offload_count": int(summary.get("total_requests") or 0),
            "local_count": 0,
            "accepted_count": int(summary.get("total_requests") or 0),
            "offload_ratio": 100.0,
            "avg_latency_ms": round(avg_latency, 1),
            "slo_ms": round(slo_ms, 1),
            "slo_pass": bool(avg_latency and avg_latency <= slo_ms),
            "online_devices": int(summary.get("connected_devices") or 0),
        },
        "server": {
            "name": "课堂分析服务器",
            "host": node,
            "port": port,
            "connected": True,
            "cpu": round(float(resources.get("cpu_util") or 0.0), 1),
            "gpu": round(float(resources.get("gpu_util") or 0.0), 1),
            "memory": round(float(resources.get("memory_util") or 0.0), 1),
            "queue": int(summary.get("queue_waiting") or 0),
        },
        "dao": {
            "strategy": "offload",
            "admission": "accept",
            "priority": priority,
            "target_node_id": "课堂分析服务器",
            "stage_count": 5,
            "stages_to_offload": ["人体/手机检测", "姿态与行为特征", "学习状态分类", "AI 教师干预"],
            "reason": "系统会结合设备能力、服务器负载和完成目标，安排当前画面的处理位置。",
        },
        "devices": devices,
        "pipeline": pipeline,
        "comparison": {
            "local_baseline_ms": baseline,
            "offload_estimate_ms": round(avg_latency, 1),
            "saved_ms": round(max(0.0, baseline - avg_latency), 1),
        },
        "events": events,
    }


def _dao_admin_degraded() -> dict:
    """:8011 不可达时 dao-admin 前端的降级结构（页面展示“等待数据”）。"""
    now = time.time()
    with _lock:
        online = sum(1 for stu in _students.values() if _is_online(stu, now))
    return {
        "connected": False,
        "timestamp": now,
        "summary": {
            "total_requests": 0,
            "offload_count": 0,
            "local_count": 0,
            "accepted_count": 0,
            "offload_ratio": 0.0,
            "avg_latency_ms": 0.0,
            "slo_ms": 200.0,
            "slo_pass": False,
            "online_devices": online,
        },
        "server": {
            "name": "课堂分析服务器",
            "host": "--",
            "port": 50051,
            "connected": False,
            "cpu": 0.0,
            "gpu": 0.0,
            "memory": 0.0,
            "queue": 0,
        },
        "dao": {
            "strategy": "waiting",
            "admission": "waiting",
            "priority": 0.0,
            "target_node_id": "--",
            "stage_count": 5,
            "stages_to_offload": [],
            "reason": "课堂分析服务器暂时没有返回调度数据，连接恢复后会自动更新。",
        },
        "devices": [],
        "pipeline": [],
        "comparison": {"local_baseline_ms": 0.0, "offload_estimate_ms": 0.0, "saved_ms": 0.0},
        "events": [],
    }


def _collect_tasks_loop(interval_sec: float = 3.0) -> None:
    """后台线程：周期采集 :8011 快照中已完成的调度任务并落库（幂等去重）。"""
    while True:
        time.sleep(interval_sec)
        snapshot = _fetch_admin_snapshot()
        if not snapshot:
            continue
        tasks = snapshot.get("recent_requests") or []
        if not tasks:
            continue
        now = time.time()
        with _lock:
            device_states = {
                device_id: stu.get("learning_state")
                for device_id, stu in _students.items()
            }
        try:
            with _history_lock, _history_conn() as conn:
                for idx, item in enumerate(tasks):
                    if item.get("status") != "completed" or not item.get("total_ms"):
                        continue
                    sequence = str(
                        item.get("sequence")
                        or f"{item.get('device_id', 'device')}-{item.get('frame_index', idx)}"
                    )
                    device_id = str(item.get("device_id") or "")
                    conn.execute(
                        "INSERT OR REPLACE INTO classroom_tasks"
                        " (sequence, device_id, frame_index, status, queue_wait_ms,"
                        "  total_ms, slo_met, learning_state, arrived_at, updated_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            sequence,
                            device_id,
                            item.get("frame_index"),
                            item.get("status"),
                            item.get("queue_wait_ms") or 0,
                            item.get("total_ms") or 0,
                            None
                            if item.get("slo_met") is None
                            else (1 if item.get("slo_met") else 0),
                            device_states.get(device_id),
                            item.get("arrived_at"),
                            now,
                        ),
                    )
        except Exception:
            pass


def _paginate(count: int, page: int, page_size: int) -> tuple[int, int, int]:
    page = max(1, int(page or 1))
    page_size = min(200, max(1, int(page_size or 20)))
    pages = max(1, math.ceil(count / page_size)) if count else 1
    return page, page_size, pages


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class StudentReport(BaseModel):
    """树莓派上报的单帧分析结果。"""

    device_id: str
    seat_label: str | None = None
    student_name: str | None = None
    learning_state: str = "unknown"
    state_label: str | None = None
    state_counts: dict[str, int] | None = None
    needs_attention: list[dict] | None = None
    frame_thumbnail: str | None = None  # data:image/jpeg;base64,...
    latency_ms: float = 0.0
    dao_strategy: str | None = None
    dao_admission: str | None = None
    edge_connected: bool = False
    timestamp: float = 0.0


class ClassroomSessionReport(BaseModel):
    """学生端在一次课堂分析结束后提交的汇总报告。"""

    device_id: str
    student_name: str | None = None
    seat_label: str | None = None
    report: dict
    timestamp: float = 0.0


class StudentOvisReview(BaseModel):
    """学生端 Ovis 事件复核结果（仅脱敏统计，不含画面）。"""

    device_id: str
    student_name: str | None = None
    candidate_state: str
    video_seconds: float = 0.0
    reviewed_at: float = 0.0
    engine: str = "ovis"
    result: dict = {}
    raw_output: str | None = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _is_online(student: dict, now: float) -> bool:
    """判断学生是否在线（最近 N 秒内有上报）。"""
    return now - student.get("last_report_ts", 0) < STUDENT_TIMEOUT_SEC


def _dominant_state(student: dict, now: float) -> str:
    """获取学生当前显示状态（离线时返回 offline）。"""
    if not _is_online(student, now):
        return "offline"
    return student.get("learning_state", "unknown")


def _cleanup_stale(now: float) -> None:
    """清理超过 5 分钟无上报的学生记录，避免列表无限增长。"""
    stale_keys = [
        key
        for key, stu in _students.items()
        if now - stu.get("last_report_ts", 0) > 300
    ]
    for key in stale_keys:
        del _students[key]


def _push_alert(device_id: str, student_name: str, state: str) -> None:
    """记录一条状态告警（去重：同一学生同一状态 30s 内只记一条）。"""
    now = time.time()
    with _lock:
        for existing in reversed(_alert_log):
            if (
                existing.get("device_id") == device_id
                and now - existing.get("timestamp", 0) < 30
            ):
                return
        _alert_log.append(
            {
                "device_id": device_id,
                "student_name": student_name,
                "state": state,
                "state_label": STATE_LABELS.get(state, state),
                "timestamp": now,
            }
        )
        if len(_alert_log) > MAX_ALERT_LOG:
            del _alert_log[: len(_alert_log) - MAX_ALERT_LOG]


def _focus_level(focus_rate: float) -> str:
    """专注等级：优秀 >=85%，良好 >=70%，一般 >=45%，其余待提升。"""
    if focus_rate >= 85:
        return "优秀"
    if focus_rate >= 70:
        return "良好"
    if focus_rate >= 45:
        return "一般"
    return "待提升"


def _event_state(event: dict) -> str:
    state = event.get("state") or event.get("learning_state") or "unknown"
    return state if state in STATE_LABELS else "unknown"


def _timeline_stats(
    events: list[dict], started_at: float, ended_at: float
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """按相邻事件差分统计各状态时长与四阶段分布（单段封顶 30 秒）。"""
    keys = STATE_ORDER + ["unknown"]
    durations = {state: 0.0 for state in keys}
    buckets = [{state: 0.0 for state in keys} for _ in range(4)]
    span = max(ended_at - started_at, 1.0)
    for index, event in enumerate(events):
        current = max(started_at, float(event.get("timestamp", started_at)))
        following = (
            float(events[index + 1].get("timestamp", ended_at))
            if index + 1 < len(events)
            else ended_at
        )
        seconds = max(0.0, min(30.0, following - current))
        state = _event_state(event)
        durations[state] += seconds
        bucket = min(3, int((current - started_at) / span * 4))
        buckets[bucket][state] += seconds
    return durations, buckets


def _fetch_session_events(started_at: float, ended_at: float) -> dict[str, list[dict]]:
    """按会话时间窗从 SQLite 历史库拉取帧事件流；库中无记录时回退内存时间线。"""
    events_by_device: dict[str, list[dict]] = {}
    try:
        with _history_conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, device_id, student_name, seat_label, learning_state"
                " FROM classroom_events"
                " WHERE timestamp >= ? AND timestamp <= ?"
                " ORDER BY timestamp ASC",
                (started_at, ended_at),
            ).fetchall()
        for row in rows:
            events_by_device.setdefault(row["device_id"], []).append(
                {
                    "timestamp": row["timestamp"],
                    "state": row["learning_state"] or "unknown",
                    "student_name": row["student_name"],
                    "seat_label": row["seat_label"],
                }
            )
    except Exception:
        pass
    if not events_by_device:
        with _lock:
            events_by_device = {
                key: list(value) for key, value in _student_timelines.items()
            }
    return events_by_device


def _build_student_report(
    device_id: str,
    events: list[dict],
    started_at: float,
    ended_at: float,
) -> dict | None:
    """生成单个学生的个人课堂报告（时长 / 专注率 / 等级 / 趋势 / 行为证据）。"""
    if not events:
        return None
    durations, buckets = _timeline_stats(events, started_at, ended_at)
    effective = sum(durations.values())
    percentages = (
        {key: round(value * 100 / effective, 1) for key, value in durations.items()}
        if effective
        else {key: 0.0 for key in durations}
    )
    focus_rate = percentages.get("focused", 0.0)
    risk_seconds = round(
        durations["distracted"] + durations["tired"] + durations["absent"], 1
    )
    stage_names = ["开场", "前段", "后段", "收尾"]
    stage_trend = []
    for index, bucket in enumerate(buckets):
        bucket_total = sum(bucket.values())
        stage_trend.append(
            {
                "stage": stage_names[index],
                "focus_rate": (
                    round(bucket["focused"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
                "distracted_rate": (
                    round(bucket["distracted"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
                "tired_rate": (
                    round(bucket["tired"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
            }
        )
    # 行为证据摘要：最长连续专注时长 + 各负面状态出现的段数与累计时长
    longest_focus = 0.0
    current_focus = 0.0
    negative_counts = {
        state: 0 for state in ("confused", "tired", "distracted", "absent")
    }
    prev_state = None
    for index, event in enumerate(events):
        state = _event_state(event)
        current = float(event.get("timestamp", started_at))
        following = (
            float(events[index + 1].get("timestamp", ended_at))
            if index + 1 < len(events)
            else ended_at
        )
        seconds = max(0.0, min(30.0, following - current))
        if state == "focused":
            current_focus += seconds
            longest_focus = max(longest_focus, current_focus)
        else:
            current_focus = 0.0
        if state in negative_counts and state != prev_state:
            negative_counts[state] += 1
        prev_state = state
    evidence = []
    if longest_focus >= 60:
        evidence.append(
            f"最长连续专注约 {int(longest_focus // 60)} 分 {int(longest_focus % 60)} 秒。"
        )
    if negative_counts["distracted"]:
        evidence.append(
            f"分心出现 {negative_counts['distracted']} 段，"
            f"累计 {round(durations['distracted'], 1)} 秒。"
        )
    if negative_counts["tired"]:
        evidence.append(
            f"疲劳出现 {negative_counts['tired']} 段，"
            f"累计 {round(durations['tired'], 1)} 秒。"
        )
    if negative_counts["confused"]:
        evidence.append(
            f"困惑出现 {negative_counts['confused']} 段，"
            f"累计 {round(durations['confused'], 1)} 秒。"
        )
    if durations["absent"] >= 10:
        evidence.append(f"离开镜头累计 {round(durations['absent'], 1)} 秒。")
    if not evidence:
        evidence.append("全程状态平稳，未出现持续负面状态。")
    suggestions = []
    if percentages.get("distracted", 0.0) >= 20:
        suggestions.append("分心时间偏多，建议上课期间收起干扰物并跟随老师节奏记笔记。")
    if percentages.get("tired", 0.0) >= 20:
        suggestions.append("疲劳占比较高，建议课前充分休息、课间适当活动。")
    if percentages.get("confused", 0.0) >= 25:
        suggestions.append("困惑时间较长，建议课中及时提问或课后复习相关知识点。")
    if percentages.get("absent", 0.0) >= 10:
        suggestions.append("离开镜头时间较长，建议保持在座位上完成整节课堂。")
    if not suggestions:
        suggestions.append("整体专注度良好，请继续保持当前学习节奏。")
    with _lock:
        stu = dict(_students.get(device_id, {}))
    first = events[0]
    return {
        "device_id": device_id,
        "student_name": stu.get("student_name")
        or first.get("student_name")
        or device_id,
        "seat_label": stu.get("seat_label") or first.get("seat_label"),
        "session_started_at": started_at,
        "session_ended_at": ended_at,
        "duration_seconds": round(effective, 1),
        "durations": {key: round(value, 1) for key, value in durations.items()},
        "state_percentages": percentages,
        "focus_rate": focus_rate,
        "focus_level": _focus_level(focus_rate),
        "risk_seconds": risk_seconds,
        "stage_trend": stage_trend,
        "evidence": evidence,
        "suggestions": suggestions,
        "engine": "classroom-statistics",
        "generated_at": ended_at,
        "limitations": "报告基于课堂期间的视觉状态统计，仅用于学习习惯观察；不用于心理、医疗或人格判断。",
    }


def _build_teacher_report(
    started_at: float,
    ended_at: float,
    events_by_device: dict[str, list[dict]] | None = None,
    student_reports: dict[str, dict] | None = None,
) -> dict:
    """汇总本次课堂期间所有学生事件与学生个人报告（全班学习分析报告）。"""
    if events_by_device is None:
        events_by_device = _fetch_session_events(started_at, ended_at)
    if student_reports is None:
        student_reports = {}
        for device_id, events in events_by_device.items():
            report = _build_student_report(device_id, events, started_at, ended_at)
            if report:
                student_reports[device_id] = report
    with _lock:
        final_reports = [
            dict(item)
            for item in _classroom_reports
            if started_at <= item.get("timestamp", 0) <= ended_at
        ]
        alerts = [
            dict(item)
            for item in _alert_log
            if started_at <= item.get("timestamp", 0) <= ended_at
        ]
        reviews = [
            dict(item)
            for values in _student_reviews.values()
            for item in values
            if started_at <= item.get("reviewed_at", 0) <= ended_at
        ]
    state_totals = {state: 0.0 for state in STATE_ORDER + ["unknown"]}
    buckets = [{state: 0.0 for state in state_totals} for _ in range(4)]
    for device_id, report in student_reports.items():
        for state, seconds in report.get("durations", {}).items():
            if state in state_totals:
                state_totals[state] += seconds
        events = events_by_device.get(device_id) or []
        _, student_buckets = _timeline_stats(events, started_at, ended_at)
        for index, bucket in enumerate(student_buckets):
            for state, seconds in bucket.items():
                buckets[index][state] += seconds
    total_state_seconds = sum(state_totals.values())
    percentages = {
        key: round(value * 100 / total_state_seconds, 1) if total_state_seconds else 0.0
        for key, value in state_totals.items()
    }
    focus_rate = percentages.get("focused", 0.0)
    focus_ranking = sorted(
        (
            {
                "device_id": report.get("device_id", device_id),
                "student_name": report.get("student_name") or device_id,
                "seat_label": report.get("seat_label"),
                "focus_rate": report.get("focus_rate", 0.0),
                "focus_level": report.get("focus_level", "待提升"),
                "risk_seconds": report.get("risk_seconds", 0.0),
                "duration_seconds": report.get("duration_seconds", 0.0),
            }
            for device_id, report in student_reports.items()
        ),
        key=lambda item: (-item["focus_rate"], item["student_name"]),
    )
    priority_students = sorted(
        focus_ranking,
        key=lambda item: (-item["risk_seconds"], item["focus_rate"]),
    )[:5]
    teacher_actions = []
    if percentages.get("distracted", 0) >= 20:
        teacher_actions.append("分心占比较高，建议增加提问、任务切换或短时互动。")
    if percentages.get("tired", 0) >= 20:
        teacher_actions.append("疲惫占比较高，建议安排短暂休息并调整讲授节奏。")
    if percentages.get("absent", 0) >= 10:
        teacher_actions.append("离开状态较多，建议核对设备位置、出勤及网络连续性。")
    low_focus = [item for item in focus_ranking if item["focus_rate"] < 45]
    if low_focus:
        teacher_actions.append(
            f"{len(low_focus)} 名学生专注率低于 45%，建议课后单独沟通并关注其座位环境。"
        )
    if not teacher_actions:
        teacher_actions.append("课堂整体状态较稳定，可保持教学节奏并持续关注个体波动。")
    trend = []
    for index, bucket in enumerate(buckets):
        bucket_total = sum(bucket.values())
        trend.append(
            {
                "stage": ["开场", "前段", "后段", "收尾"][index],
                "focus_rate": (
                    round(bucket["focused"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
                "distracted_rate": (
                    round(bucket["distracted"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
                "tired_rate": (
                    round(bucket["tired"] * 100 / bucket_total, 1)
                    if bucket_total
                    else 0.0
                ),
            }
        )
    return {
        "engine": "classroom-statistics",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(ended_at - started_at, 1),
        "student_count": len(student_reports),
        "completed_student_reports": len(student_reports),
        "focus_rate": focus_rate,
        "state_durations": {key: round(value, 1) for key, value in state_totals.items()},
        "state_percentages": percentages,
        "attention_student_count": sum(
            1
            for report in student_reports.values()
            if report.get("risk_seconds", 0.0)
            > report.get("duration_seconds", 0.0) * 0.35
        ),
        "priority_students": priority_students,
        "focus_ranking": focus_ranking,
        "stage_trend": trend,
        "alert_count": len(alerts),
        "ovis_review_count": len(reviews),
        "ovis_disagreement_count": sum(
            1
            for item in reviews
            if item.get("result", {}).get("verified_state")
            not in {item.get("candidate_state"), "unknown", None}
        ),
        "teacher_actions": teacher_actions,
        "summary": (
            f"本次课堂覆盖 {len(student_reports)} 名学生，整体专注度 {focus_rate:.1f}%，"
            f"已生成 {len(student_reports)} 份学生个人课堂报告。"
        ),
        "limitations": "报告按设备采样和视觉状态汇总，仅用于教学过程观察；不用于心理、医疗或人格判断。",
    }


def _generate_reports_job(started_at: float, ended_at: float) -> None:
    """下课后先发布统计报告，再异步等待 Ovis 增强并原位更新。"""
    try:
        events_by_device = _fetch_session_events(started_at, ended_at)
        student_reports: dict[str, dict] = {}
        for device_id, events in events_by_device.items():
            report = _build_student_report(device_id, events, started_at, ended_at)
            if report:
                student_reports[device_id] = report
        class_report = _build_teacher_report(
            started_at, ended_at, events_by_device, student_reports
        )
        # 统计聚合很快，先让教师和学生立即看到完整的基础报告。Ovis 模型加载与
        # 逐份生成可能耗时较长，不应让整个报告页面一直空白等待。
        class_report["ovis_status"] = "generating"
        with _lock:
            for device_id, report in student_reports.items():
                report["ovis_status"] = "generating"
                _student_final_reports[device_id] = report
            _teacher_session["report"] = class_report
            _teacher_session["generating"] = False

        polished_bundle = _try_ovis_polish(
            {
                "report_kind": "classroom_bundle",
                "class_report": class_report,
                "student_reports": student_reports,
            }
        )
        bundle_class = polished_bundle.get("class_report") or {}
        if (
            polished_bundle.get("report_kind") == "classroom_bundle"
            and bundle_class.get("engine") == "classroom-statistics-ovis"
            and bundle_class.get("ovis_report")
        ):
            class_report = polished_bundle.get("class_report") or class_report
            student_reports = polished_bundle.get("student_reports") or student_reports
        else:
            # Ovis 不可用时保留完整规则报告，并把降级原因放到全班报告中展示。
            class_report.update(
                {
                    key: polished_bundle[key]
                    for key in ("ovis_status", "ovis_skip_reason", "ovis_error")
                    if key in polished_bundle
                }
            )
            class_report.pop("ovis_status", None)
            # _try_ovis_polish 会补回明确的 skipped/failed 状态；若没有则保持降级。
            for key in ("ovis_status", "ovis_skip_reason", "ovis_error"):
                if key in polished_bundle:
                    class_report[key] = polished_bundle[key]
            # 子进程失败时不能让学生端永远停留在“正在生成”。
            for report in student_reports.values():
                for key in ("ovis_status", "ovis_skip_reason", "ovis_error"):
                    if key in polished_bundle:
                        report[key] = polished_bundle[key]
        with _lock:
            for device_id, report in student_reports.items():
                _student_final_reports[device_id] = report
            _teacher_session["report"] = class_report
            _teacher_session["generating"] = False
        for device_id, report in student_reports.items():
            _record_report(
                {
                    "device_id": device_id,
                    "student_name": report.get("student_name"),
                    "seat_label": report.get("seat_label"),
                    "engine": report.get("engine", "classroom-statistics"),
                    "timestamp": ended_at,
                    "report": report,
                }
            )
        # 教师全班总结同样写入 SQLite 报告流，供历史页追溯
        _record_report(
            {
                "device_id": "teacher-summary",
                "student_name": "教师总结",
                "seat_label": "-",
                "engine": class_report.get("engine", "classroom-statistics"),
                "timestamp": ended_at,
                "report": class_report,
            }
        )
    except Exception as exc:  # 生成失败：记录原因并降级（不影响已统计状态）
        with _lock:
            _teacher_session["generating"] = False
            _teacher_session["error"] = f"报告生成失败：{exc}"


def _candidate_ovis_model_dirs() -> list[Path]:
    """Ovis2-2B-GPTQ-Int4 模型目录候选：环境变量优先，其次常见落盘位置。"""
    candidates: list[Path] = []
    env_dir = os.getenv("CLASSROOM_OVIS_MODEL_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    server_root = BASE_DIR.parent  # experiment_server/
    challenge_root = BASE_DIR.parents[2]  # Challenge-cup/Challenge-cup/
    workspace_root = Path(
        os.getenv("CLASSROOM_WORKSPACE_ROOT", "D:/WORK/Project")
    )
    candidates.extend(
        [
            BASE_DIR / "data" / "models" / "Ovis2-2B-GPTQ-Int4",
            challenge_root / "industrial-inspection" / "Ovis2-2B-GPTQ-Int4",
            server_root / "data" / "models" / "industrial" / "Ovis2-2B-GPTQ-Int4",
            server_root / "data" / "models" / "Ovis2-2B-GPTQ-Int4",
            workspace_root / "industrial-inspection" / "Ovis2-2B-GPTQ-Int4",
            workspace_root / "industrial-inspection" / "model" / "Ovis2-2B-GPTQ-Int4",
            workspace_root
            / "industrial-inspection"
            / "industrial-inspection"
            / "Ovis2-2B-GPTQ-Int4",
            workspace_root
            / "industrial-inspection"
            / "industrial-inspection"
            / "model"
            / "Ovis2-2B-GPTQ-Int4",
            Path.home() / "models" / "Ovis2-2B-GPTQ-Int4",
        ]
    )
    return candidates


def _resolve_ovis_model_dir() -> Path | None:
    """返回第一个含 config.json 且含 safetensors 权重的候选目录。"""
    for directory in _candidate_ovis_model_dirs():
        try:
            if (directory / "config.json").exists() and any(
                directory.glob("*.safetensors")
            ):
                return directory
        except OSError:
            continue
    return None


def _candidate_ovis_pythons() -> list[str]:
    """带 gptqmodel 的解释器候选：环境变量优先，其次 challenge-cup-ovis conda 环境。"""
    candidates: list[Path] = []
    env_python = os.getenv("CLASSROOM_OVIS_PYTHON", "").strip()
    if env_python:
        candidates.append(Path(env_python))
    conda_roots = [
        Path.home() / "miniconda3",
        Path.home() / ".conda",
        Path.home() / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/Anaconda3"),
    ]
    for root in conda_roots:
        candidates.append(root / "envs" / "challenge-cup-ovis" / "python.exe")
    # 仅当当前解释器确实装有 gptqmodel 时才作为候选。此前 edu-server
    # 被盲目加入，覆盖了 challenge-cup-ovis 的真实错误，最终只显示
    # “No module named gptqmodel”。
    if importlib.util.find_spec("gptqmodel") is not None:
        candidates.append(Path(sys.executable))
    return [str(item) for item in candidates if item.exists()]


def _try_ovis_polish(report: dict) -> dict:
    """尝试用 classroom_report_ovis.py 子进程对规则报告做 Ovis2 润色。

    模型目录与解释器均按候选路径自动探测（环境变量可覆盖，参照工业端
    Ovis 探测顺序）；任何失败（环境缺失、超时、模型加载错误）都降级为
    规则统计报告，并把原因写入 ovis_status / ovis_skip_reason /
    ovis_error，不影响下课流程。
    """
    script = BASE_DIR / "classroom_report_ovis.py"
    if not script.exists():
        report["ovis_status"] = "skipped"
        report["ovis_skip_reason"] = "缺少 classroom_report_ovis.py"
        return report
    model_dir = _resolve_ovis_model_dir()
    if model_dir is None:
        report["ovis_status"] = "skipped"
        report["ovis_skip_reason"] = (
            "未找到 Ovis2-2B-GPTQ-Int4 模型目录（可设置 CLASSROOM_OVIS_MODEL_DIR）"
        )
        return report
    python_candidates = _candidate_ovis_pythons()
    if not python_candidates:
        report["ovis_status"] = "skipped"
        report["ovis_skip_reason"] = (
            "未找到可用的 Ovis 解释器（可设置 CLASSROOM_OVIS_PYTHON）"
        )
        return report
    last_error = ""
    for python_bin in python_candidates:
        try:
            with tempfile.TemporaryDirectory(prefix="classroom-report-") as temp_dir:
                report_path = Path(temp_dir) / "rule_report.json"
                out_path = Path(temp_dir) / "ovis_report.json"
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False), encoding="utf-8"
                )
                env = {**os.environ, "CLASSROOM_OVIS_MODEL_DIR": str(model_dir)}
                completed = subprocess.run(
                    [
                        python_bin,
                        str(script),
                        "--report-json",
                        str(report_path),
                        "--out-json",
                        str(out_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=float(os.getenv("CLASSROOM_OVIS_TIMEOUT", "480")),
                    env=env,
                    cwd=str(BASE_DIR),
                )
                if completed.returncode == 0 and out_path.exists():
                    polished = json.loads(out_path.read_text(encoding="utf-8"))
                    if isinstance(polished, dict) and (
                        polished.get("ovis_report")
                        or polished.get("report_kind") == "classroom_bundle"
                    ):
                        return polished
                stdout_text = (completed.stdout or "").strip()
                stderr_text = (completed.stderr or "").strip()
                # 模型依赖会把大量无害 warning 写到 stderr；真正异常由脚本以
                # JSON 写到 stdout。优先提取它，避免页面只显示警告的尾巴。
                actual_error = ""
                for line in reversed(stdout_text.splitlines()):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("error"):
                        actual_error = str(payload["error"])
                        break
                last_error = (actual_error or stdout_text or stderr_text)
                last_error = last_error.strip().replace("\n", " ")[-800:]
        except Exception as exc:
            last_error = str(exc)[:500]
    report["ovis_status"] = "failed"
    report["ovis_error"] = last_error or "Ovis 子进程未返回有效结果"
    return report


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="线下课堂教师管理端", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if CLASSROOM_WEB_DIR.exists():
    app.mount(
        "/classroom-assets",
        StaticFiles(directory=CLASSROOM_WEB_DIR),
        name="classroom_assets",
    )

if ADMIN_FRONTEND_DIR.exists():
    app.mount(
        "/admin-assets",
        StaticFiles(directory=ADMIN_FRONTEND_DIR),
        name="admin_assets",
    )

if VUE_FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=VUE_FRONTEND_DIR),
        name="vue_assets",
    )


# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/console")
async def console_page():
    """教师端入口：直接进入教师总览页。

    独立教育栈不需要额外的管理端壳页，/ 与 /console 一并指向教师总览。
    """
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/dashboard")
async def dashboard_page():
    """教师总览页（作为管控台 iframe 的默认内容）。"""
    index = CLASSROOM_WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "前端文件未找到"}, status_code=404)


@app.get("/admin")
async def admin_page():
    """8008 只提供课堂数据服务；调度管理页面统一由 8009 提供。"""
    return RedirectResponse("http://127.0.0.1:8009/admin", status_code=302)

# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.post("/api/student/report")
async def receive_report(report: StudentReport):
    """接收树莓派上报的学生学习状态。"""
    now = time.time()
    ts = report.timestamp or now
    state_label = report.state_label or STATE_LABELS.get(
        report.learning_state, report.learning_state
    )

    student_entry = {
        "device_id": report.device_id,
        "seat_label": report.seat_label or "未命名座位",
        "student_name": report.student_name or "未命名学生",
        "learning_state": report.learning_state,
        "state_label": state_label,
        "state_counts": report.state_counts or {},
        "needs_attention": report.needs_attention or [],
        "frame_thumbnail": report.frame_thumbnail,
        "latency_ms": round(report.latency_ms, 1),
        "dao_strategy": report.dao_strategy or "unknown",
        "dao_admission": report.dao_admission,
        "edge_connected": report.edge_connected,
        "last_report_ts": ts,
        "report_count": 0,
    }

    prev_state: str | None = None
    with _lock:
        if report.device_id in _students:
            prev_entry = _students[report.device_id]
            prev_state = prev_entry.get("learning_state")
            student_entry["report_count"] = prev_entry.get("report_count", 0) + 1
            student_entry["alerted_negative"] = prev_entry.get("alerted_negative", False)
        else:
            student_entry["alerted_negative"] = False
        # 缺勤连续计数：抑制网络抖动/回退本地导致的瞬时缺勤刷屏
        prev_absent_streak = (
            prev_entry.get("absent_streak", 0)
            if report.device_id in _students
            else 0
        )
        student_entry["absent_streak"] = (
            prev_absent_streak + 1 if report.learning_state == "absent" else 0
        )
        _students[report.device_id] = student_entry
        # 学生状态时间线：供个人报告 / 教师汇总报告按持续时长统计
        timeline = _student_timelines.setdefault(report.device_id, [])
        timeline.append(
            {"timestamp": ts, "state": report.learning_state, "label": state_label}
        )
        del timeline[:-MAX_TIMELINE_EVENTS]
        _cleanup_stale(now)

    absent_streak = student_entry["absent_streak"]
    already_alerted = student_entry.get("alerted_negative", False)

    # 学生恢复非负面状态：清除历史告警 + 重置告警标志（允许下次负面状态再告警）
    if report.learning_state not in ATTENTION_STATES:
        with _lock:
            _alert_log[:] = [
                a for a in _alert_log if a.get("device_id") != report.device_id
            ]
            if report.device_id in _students:
                _students[report.device_id]["alerted_negative"] = False
    # 负面状态：同一负面状态持续期间只告警一次（避免同状态刷屏）；
    # 但负面状态发生切换（如换视频后 疲惫→分心）时，用新状态告警替换旧告警，
    # 保证教师总览的实时告警始终反映学生最新状态。
    # - absent 需连续3次才触发（网络抖动不告警）
    # - 其他负面状态即时触发
    else:
        state_changed = prev_state is not None and prev_state != report.learning_state
        if not already_alerted or state_changed:
            should_alert = report.learning_state != "absent" or absent_streak >= 3
            if should_alert:
                if state_changed:
                    # 先移除该学生的旧状态告警，让新状态告警顶替
                    with _lock:
                        _alert_log[:] = [
                            a
                            for a in _alert_log
                            if a.get("device_id") != report.device_id
                        ]
                _push_alert(
                    report.device_id,
                    report.student_name or report.seat_label or report.device_id,
                    report.learning_state,
                )
                with _lock:
                    if report.device_id in _students:
                        _students[report.device_id]["alerted_negative"] = True

    # —— 课堂历史持久化：事件 + 报告（供课堂中心页 / 调度页追溯） ——
    report_entry = {
        "device_id": report.device_id,
        "seat_label": report.seat_label or "未命名座位",
        "student_name": report.student_name or "未命名学生",
        "learning_state": report.learning_state,
        "state_label": state_label,
        "state_counts": report.state_counts or {},
        "needs_attention": report.needs_attention or [],
        "latency_ms": round(report.latency_ms, 1),
        "dao_strategy": report.dao_strategy or "unknown",
        "dao_admission": report.dao_admission,
        "edge_connected": report.edge_connected,
        "engine": "边缘推理" if report.edge_connected else "本地推理",
        "timestamp": ts,
    }
    _record_event(report_entry)
    _record_report(report_entry)
    with _session_lock:
        if _session_state["started_at"] is None:
            _session_state["started_at"] = ts
        _session_state["last_report_at"] = ts
        _session_state["ended_at"] = None

    return {"ok": True, "online_students": len(_students)}


@app.get("/api/students")
async def get_students():
    """返回所有学生的当前状态（供前端轮询）。"""
    now = time.time()
    with _lock:
        _cleanup_stale(now)
        students_list = []
        for stu in _students.values():
            online = _is_online(stu, now)
            display_state = stu.get("learning_state", "unknown") if online else "offline"
            students_list.append(
                {
                    **stu,
                    "online": online,
                    "display_state": display_state,
                    "display_label": STATE_LABELS.get(display_state, display_state),
                    "seconds_since_report": round(now - stu.get("last_report_ts", 0), 1),
                }
            )

    # 按座位号排序
    students_list.sort(key=lambda s: (not s["online"], s.get("seat_label", "")))
    return {"students": students_list, "server_time": now}


@app.get("/api/summary")
async def get_summary():
    """返回课堂整体统计摘要。"""
    now = time.time()
    with _lock:
        _cleanup_stale(now)
        state_summary = {state: 0 for state in STATE_ORDER}
        state_summary["offline"] = 0
        state_summary["unknown"] = 0
        # 按实际上报时间累计各状态持续时长；单段最多 30 秒，避免设备断开后
        # 把离线时间误计入某个学习状态。
        state_duration_seconds = {state: 0.0 for state in state_summary}
        for events in _student_timelines.values():
            for current, following in zip(events, events[1:] + [{"timestamp": now}]):
                state = current.get("state", "unknown")
                if state not in state_duration_seconds:
                    state = "unknown"
                seconds = max(
                    0.0, min(30.0, following["timestamp"] - current["timestamp"])
                )
                state_duration_seconds[state] += seconds
        total = 0
        online_count = 0
        attention_count = 0

        for stu in _students.values():
            total += 1
            online = _is_online(stu, now)
            if online:
                online_count += 1
            display_state = stu.get("learning_state", "unknown") if online else "offline"
            if display_state in state_summary:
                state_summary[display_state] += 1
            else:
                state_summary["unknown"] += 1
            if display_state in ATTENTION_STATES:
                attention_count += 1

        avg_latency = 0.0
        online_latencies = [
            stu["latency_ms"]
            for stu in _students.values()
            if _is_online(stu, now) and stu.get("latency_ms", 0) > 0
        ]
        if online_latencies:
            avg_latency = round(sum(online_latencies) / len(online_latencies), 1)

    return {
        "total_students": total,
        "online_students": online_count,
        "attention_students": attention_count,
        "state_summary": state_summary,
        "state_duration_seconds": {
            state: round(seconds, 1) for state, seconds in state_duration_seconds.items()
        },
        "state_labels": STATE_LABELS,
        "avg_latency_ms": avg_latency,
        "server_time": now,
    }


@app.get("/api/alerts")
async def get_alerts():
    """返回最近的状态告警记录。"""
    now = time.time()
    with _lock:
        alerts = [
            {
                **a,
                "seconds_ago": round(now - a.get("timestamp", 0), 1),
            }
            for a in reversed(_alert_log[-50:])
        ]
    return {"alerts": alerts}


@app.post("/api/clear")
async def clear_all():
    """清空所有学生数据（测试 / 重置用）。"""
    with _lock:
        _students.clear()
        _alert_log.clear()
        _student_timelines.clear()
        _student_reviews.clear()
        _classroom_reports.clear()
        _student_final_reports.clear()
        _teacher_session.update(
            {
                "active": False,
                "started_at": None,
                "ended_at": None,
                "report": None,
                "generating": False,
                "error": None,
            }
        )
    with _session_lock:
        _session_state.update(
            {"started_at": None, "last_report_at": None, "ended_at": None}
        )
    return {"ok": True}


# ---------------------------------------------------------------------------
# 挑战杯课堂中心页 / 调度页 数据接口（CORS 已全局放开）
# ---------------------------------------------------------------------------

@app.get("/api/admin/snapshot")
async def admin_snapshot():
    """DAO 调度快照：优先转发教育 gRPC 管理端(:8011)，失败时降级为本服务聚合。"""
    # urlopen 是同步阻塞调用。直接放在 async 路由里会每3秒冻结整个8008
    # 最多2.5秒，导致课堂中心并发请求随机超时。放入工作线程后，其余课堂
    # 接口仍能立即响应。
    snap = await asyncio.to_thread(_fetch_admin_snapshot)
    if snap is not None:
        snap.setdefault("connected", True)
        for idx, item in enumerate(snap.get("recent_requests") or []):
            if not item.get("sequence"):
                item["sequence"] = idx + 1
        return snap
    now = time.time()
    with _lock:
        latencies = [
            stu.get("latency_ms", 0)
            for stu in _students.values()
            if _is_online(stu, now) and stu.get("latency_ms", 0) > 0
        ]
        online = sum(1 for stu in _students.values() if _is_online(stu, now))
        total = len(_students)
    return {
        "connected": False,
        "timestamp": now,
        "summary": {
            "queue_waiting": 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "connected_devices": total,
            "realtime_devices": online,
            "slo_rate": 0.0,
        },
        "resources": {
            "resource_state": "edge-offline",
            "gpu_util": 0.0,
            "cpu_util": 0.0,
            "idle_workers": None,
            "max_workers": None,
        },
        "dao": {"policy": "waiting", "slo_ms": 0.0},
        "workflow": [],
        "devices": [],
        "recent_requests": [],
        "schedule_order": [],
    }


@app.get("/api/admin/dao-snapshot")
async def admin_dao_snapshot():
    """dao-admin 前端专用：:8011 全量快照 → dao-admin 格式；不可达时降级。

    与 /api/admin/snapshot（保留 :8011 原始格式）不同，本端点做字段适配，
    供 8008 /admin 页面轮询。
    """
    snap = _fetch_admin_snapshot()
    if snap is not None:
        return _adapt_admin_snapshot(snap)
    return _dao_admin_degraded()


@app.post("/api/teacher/session/start")
async def start_teacher_session():
    """教师点击「开始上课」：开启手动会话并记录起点，清空上一节课报告。"""
    now = time.time()
    with _lock:
        _teacher_session.update(
            {
                "active": True,
                "started_at": now,
                "ended_at": None,
                "report": None,
                "generating": False,
                "error": None,
            }
        )
        _student_final_reports.clear()
    with _session_lock:
        _session_state["started_at"] = now
        _session_state["last_report_at"] = now
        _session_state["ended_at"] = None
    return {"ok": True, "started_at": now}


@app.post("/api/teacher/session/finish")
async def finish_teacher_session():
    """教师点击「下课 · 生成报告」：进入「正在生成报告」状态，后台线程异步生成。

    前端通过 GET /api/teacher/session 轮询 generating 字段：
    报告生成完成后 report 字段自动就位；失败时 error 记录原因。
    """
    ended_at = time.time()
    with _lock:
        started_at = _teacher_session.get("started_at")
        if not started_at or not _teacher_session.get("active"):
            return JSONResponse({"error": "课堂尚未开始或已下课"}, status_code=409)
        _teacher_session.update(
            {
                "active": False,
                "ended_at": ended_at,
                "report": None,
                "generating": True,
                "error": None,
            }
        )
    with _session_lock:
        _session_state["ended_at"] = ended_at
    threading.Thread(
        target=_generate_reports_job,
        args=(float(started_at), ended_at),
        daemon=True,
    ).start()
    return {"ok": True, "generating": True, "ended_at": ended_at}


@app.get("/api/teacher/session")
async def teacher_session():
    """返回教师手动控制的课堂会话；学生持续分析不会自动触发开课。"""
    with _lock:
        manual = dict(_teacher_session)
    return {
        "active": bool(manual.get("active")),
        "started_at": manual.get("started_at"),
        "ended_at": manual.get("ended_at"),
        "report": manual.get("report"),
        "generating": bool(manual.get("generating")),
        "error": manual.get("error"),
        "last_report_at": None,
    }


@app.post("/api/classroom/report")
async def receive_classroom_report(payload: ClassroomSessionReport):
    """保存学生端生成的最终课堂报告；不接收原始视频或图片。"""
    item = {
        "device_id": payload.device_id,
        "student_name": payload.student_name or "未命名学生",
        "seat_label": payload.seat_label or "未命名座位",
        "report": payload.report,
        "timestamp": payload.timestamp or time.time(),
    }
    with _lock:
        _classroom_reports.insert(0, item)
        del _classroom_reports[MAX_CLASSROOM_REPORTS:]
    _record_report(
        {
            "device_id": item["device_id"],
            "student_name": item["student_name"],
            "seat_label": item["seat_label"],
            "engine": (item.get("report") or {}).get("engine", "规则汇总"),
            "timestamp": item["timestamp"],
            "report": item["report"],
        }
    )
    return {"ok": True}


@app.post("/api/student/review")
async def receive_student_review(review: StudentOvisReview):
    """保存学生端 Ovis 事件复核结果（脱敏统计）。"""
    item = review.model_dump()
    item["reviewed_at"] = review.reviewed_at or time.time()
    with _lock:
        values = _student_reviews.setdefault(review.device_id, [])
        values.append(item)
        del values[:-50]
        if review.device_id in _students:
            _students[review.device_id]["latest_ovis_review"] = item
    return {"ok": True}


@app.get("/api/students/{device_id}/reviews")
async def get_student_reviews(device_id: str):
    with _lock:
        return {"reviews": list(_student_reviews.get(device_id, []))}


@app.get("/api/students/{device_id}/timeline")
async def get_student_timeline(device_id: str):
    """返回单个学生的状态时间线与各状态持续时长。"""
    with _lock:
        items = list(_student_timelines.get(device_id, []))
    durations: dict[str, float] = {}
    for current, following in zip(items, items[1:] + [{"timestamp": time.time()}]):
        seconds = max(0.0, min(30.0, following["timestamp"] - current["timestamp"]))
        durations[current["state"]] = durations.get(current["state"], 0.0) + seconds
    return {
        "events": items,
        "durations_seconds": {
            key: round(value, 1) for key, value in durations.items()
        },
    }


@app.get("/api/students/{device_id}/report")
async def get_student_report(device_id: str):
    """只返回下课后生成完成的个人报告，不在上课期间临时生成报告。

    视频分析和状态上报始终运行；教师点击开始上课只划定报告统计时间窗。
    学生端可根据 session 状态区分未开课、上课中和报告生成中。
    """
    with _lock:
        final = dict(_student_final_reports.get(device_id) or {}) or None
        session_snapshot = {
            "active": bool(_teacher_session.get("active")),
            "generating": bool(_teacher_session.get("generating")),
            "ended_at": _teacher_session.get("ended_at"),
        }
    if final:
        final["final"] = True
        final["session"] = session_snapshot
        return final
    if session_snapshot["active"]:
        message = "课堂进行中，个人报告将在教师下课后生成"
    elif session_snapshot["generating"]:
        message = "课堂已结束，正在生成个人报告"
    elif session_snapshot["ended_at"]:
        message = "本次课堂没有可用的个人报告"
    else:
        message = "教师尚未开始上课，暂无个人课堂报告"
    return JSONResponse(
        {"error": message, "report_ready": False, "session": session_snapshot},
        status_code=409,
    )


@app.get("/api/history/summary")
async def history_summary():
    """历史课堂统计（SQLite 聚合），供课堂中心页与调度页共用。"""
    try:
        with _history_conn() as conn:
            total_events = conn.execute(
                "SELECT COUNT(*) FROM classroom_events"
            ).fetchone()[0]
            state_counts = {
                row["learning_state"] or "unknown": row["cnt"]
                for row in conn.execute(
                    "SELECT learning_state, COUNT(*) AS cnt FROM classroom_events"
                    " GROUP BY learning_state"
                )
            }
            attention_events = conn.execute(
                "SELECT COUNT(*) FROM classroom_events WHERE learning_state IN"
                " ('distracted','tired','absent')"
            ).fetchone()[0]
            task_row = conn.execute(
                "SELECT COUNT(*) AS total, AVG(total_ms) AS avg_ms,"
                " SUM(CASE WHEN slo_met = 1 THEN 1 ELSE 0 END) AS met"
                " FROM classroom_tasks"
            ).fetchone()
    except Exception:
        total_events, state_counts, attention_events = 0, {}, 0
        task_row = None
    total_tasks = task_row["total"] if task_row else 0
    attention_rate = (
        round(attention_events * 100.0 / total_events, 1) if total_events else 0.0
    )
    return {
        "scene": "classroom",
        "state_counts": state_counts,
        "total_events": total_events,
        "attention_rate": attention_rate,
        "accuracy_pct": None,
        "detection_accuracy_pct": None,
        "total_tasks": total_tasks,
        "avg_task_latency_ms": (
            round(task_row["avg_ms"], 1) if task_row and task_row["avg_ms"] else 0.0
        ),
        "slo_rate": (
            round(task_row["met"] * 100.0 / total_tasks, 1)
            if task_row and total_tasks and task_row["met"] is not None
            else 0.0
        ),
    }


@app.get("/api/history/events")
async def history_events(page: int = 1, page_size: int = 30):
    """分页返回历史课堂帧事件（学生上报逐条记录）。"""
    total = 0
    rows: list[sqlite3.Row] = []
    pages = 1
    try:
        with _history_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM classroom_events"
            ).fetchone()[0]
            page, page_size, pages = _paginate(total, page, page_size)
            rows = conn.execute(
                "SELECT * FROM classroom_events ORDER BY timestamp DESC, id DESC"
                " LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size),
            ).fetchall()
    except Exception:
        return {"items": [], "page": page, "pages": 1, "total": 0}
    items = [
        {
            "timestamp": row["timestamp"],
            "device_id": row["device_id"],
            "student_name": row["student_name"],
            "seat_label": row["seat_label"],
            "learning_state": row["learning_state"],
            "state_label": row["state_label"],
            "latency_ms": row["latency_ms"],
            "dao_strategy": row["dao_strategy"],
            "edge_connected": bool(row["edge_connected"]),
        }
        for row in rows
    ]
    return {"items": items, "page": page, "pages": pages, "total": total}


@app.get("/api/history/tasks")
async def history_tasks(page: int = 1, page_size: int = 30):
    """分页返回历史课堂调度任务（来自 :8011 快照的已完成请求）。"""
    total = 0
    rows: list[sqlite3.Row] = []
    pages = 1
    try:
        with _history_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM classroom_tasks"
            ).fetchone()[0]
            page, page_size, pages = _paginate(total, page, page_size)
            rows = conn.execute(
                "SELECT * FROM classroom_tasks"
                " ORDER BY COALESCE(arrived_at, updated_at) DESC"
                " LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size),
            ).fetchall()
    except Exception:
        return {"items": [], "page": page, "pages": 1, "total": 0}
    items = [
        {
            "sequence": row["sequence"],
            "device_id": row["device_id"],
            "frame_index": row["frame_index"],
            "status": row["status"],
            "queue_wait_ms": row["queue_wait_ms"],
            "total_ms": row["total_ms"],
            "slo_met": None if row["slo_met"] is None else bool(row["slo_met"]),
            "learning_state": row["learning_state"],
        }
        for row in rows
    ]
    return {"items": items, "page": page, "pages": pages, "total": total}


@app.get("/api/classroom/reports")
async def classroom_reports(page: int = 1, page_size: int = 20):
    """分页返回学生端课堂报告历史。"""
    total = 0
    rows: list[sqlite3.Row] = []
    pages = 1
    try:
        with _history_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM classroom_reports"
            ).fetchone()[0]
            page, page_size, pages = _paginate(total, page, page_size)
            rows = conn.execute(
                "SELECT * FROM classroom_reports ORDER BY timestamp DESC, id DESC"
                " LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size),
            ).fetchall()
    except Exception:
        return {"reports": [], "page": page, "pages": 1, "total": 0}
    reports = [
        {
            "timestamp": row["timestamp"],
            "device_id": row["device_id"],
            "student_name": row["student_name"],
            "seat_label": row["seat_label"],
            "report": {"engine": row["engine"] or "规则汇总"},
        }
        for row in rows
    ]
    return {"reports": reports, "page": page, "pages": pages, "total": total}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="线下课堂教师管理端服务")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8008, help="绑定端口")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("  线下课堂教师管理端 (管理端)")
    print(f"  监听地址: http://{args.host}:{args.port}/")
    print(f"  管控台:   http://<本机IP>:{args.port}/  (教师总览 + DAO监控)")
    print(f"  上报接口: POST http://<本机IP>:{args.port}/api/student/report")
    print(f"  数据查询: GET  http://<本机IP>:{args.port}/api/students")
    print(f"{'=' * 60}")

    _init_history_db()
    threading.Thread(
        target=_collect_tasks_loop, name="classroom-task-collector", daemon=True
    ).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
