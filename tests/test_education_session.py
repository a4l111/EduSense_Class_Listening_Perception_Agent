import asyncio
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


EDUCATION_DIR = Path(__file__).resolve().parents[1]


def load_dashboard(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "education_classroom_dashboard_test", EDUCATION_DIR / "classroom_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HISTORY_DB_PATH = tmp_path / "classroom.db"
    module._init_history_db()
    module._resolve_ovis_model_dir = lambda: None
    return module


class EducationSessionTests(unittest.TestCase):
    def test_reports_only_exist_after_teacher_finishes_class(self):
      # 后台 SQLite 写线程在 Windows 上可能短暂持有 WAL 文件，测试退出时忽略清理竞争。
      with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        dashboard = load_dashboard(Path(temp_dir))
        with TestClient(dashboard.app) as client:
        # 没有开始上课时仍接收实时分析，但教师页不会把学生卡片作为课堂展示。
          first = client.post(
            "/api/student/report",
            json={
                "device_id": "pi-01",
                "student_name": "学生一",
                "seat_label": "A1",
                "learning_state": "focused",
                "latency_ms": 80,
                "edge_connected": True,
            },
        )
          self.assertEqual(first.status_code, 200)
          self.assertEqual(client.get("/api/students/pi-01/report").status_code, 409)
          self.assertFalse(client.get("/api/teacher/session").json()["active"])

          self.assertEqual(client.post("/api/teacher/session/start").status_code, 200)
          for device_id, state in (("pi-01", "focused"), ("pi-02", "distracted")):
            response = client.post(
                "/api/student/report",
                json={
                    "device_id": device_id,
                    "student_name": device_id,
                    "seat_label": device_id,
                    "learning_state": state,
                    "latency_ms": 90,
                    "edge_connected": True,
                },
            )
            self.assertEqual(response.status_code, 200)

          live = client.get("/api/students/pi-01/report")
          self.assertEqual(live.status_code, 409)
          self.assertTrue(live.json()["session"]["active"])

          finish = client.post("/api/teacher/session/finish")
          self.assertEqual(finish.status_code, 200)
          deadline = time.time() + 5
          session = client.get("/api/teacher/session").json()
          while session["generating"] and time.time() < deadline:
              time.sleep(0.05)
              session = client.get("/api/teacher/session").json()
          self.assertFalse(session["generating"])
          self.assertEqual(session["report"]["student_count"], 2)
          personal = client.get("/api/students/pi-01/report")
          self.assertEqual(personal.status_code, 200)
          self.assertTrue(personal.json()["final"])

        # 下一节课开始时刷新上一节报告。
          self.assertEqual(client.post("/api/teacher/session/start").status_code, 200)
          refreshed = client.get("/api/students/pi-01/report")
          self.assertEqual(refreshed.status_code, 409)
          self.assertTrue(refreshed.json()["session"]["active"])


    def test_8009_admin_prefers_full_classroom_snapshot(self):
      sys.path.insert(0, str(EDUCATION_DIR))
      import web_api

      expected = {
        "connected": True,
        "devices": [{"id": f"pi-0{i}", "name": f"用户 pi-0{i}"} for i in range(1, 4)],
        "events": [{"id": "one", "timestamp": time.time(), "latency_ms": 88}],
    }
      with patch.object(web_api, "classroom_admin_snapshot", return_value=expected):
        with TestClient(web_api.app) as client:
          response = client.get("/api/admin/dao-snapshot")
      self.assertEqual(response.status_code, 200)
      self.assertEqual(len(response.json()["devices"]), 3)

    def test_admin_snapshot_keeps_real_stage_timing_for_three_charts(self):
      with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        dashboard = load_dashboard(Path(temp_dir))
        now = time.time()
        source = {
          "summary": {"total_requests": 1, "avg_latency_ms": 86, "connected_devices": 1},
          "server": {"node_id": "edge-01", "grpc_port": 50051},
          "resources": {},
          "dao": {},
          "devices": [{"device_id": "client_pi-01", "total_requests": 1, "last_seen_age_sec": 1}],
          "recent_requests": [{
            "request_id": "req-1", "sequence": 1, "device_id": "client_pi-01",
            "arrived_at": now, "frame_index": 9, "status": "completed", "total_ms": 86,
            "actual_schedule": [
              {"stage": "YOLO Detect", "actual_start": now, "actual_end": now + 0.03, "duration_ms": 30},
              {"stage": "YOLO-pose", "actual_start": now + 0.03, "actual_end": now + 0.07, "duration_ms": 40},
            ],
            "stage_timings": {"YOLO": 30, "Pose": 40},
          }],
        }
        adapted = dashboard._adapt_admin_snapshot(source)
        self.assertEqual(adapted["events"][0]["device_id"], "pi-01")
        self.assertEqual(len(adapted["events"][0]["actual_schedule"]), 2)
        self.assertEqual(adapted["events"][0]["stage_timings"]["Pose"], 40)

    def test_admin_snapshot_uses_newest_requests_from_appendleft_feed(self):
      with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        dashboard = load_dashboard(Path(temp_dir))
        now = time.time()
        # 8011 的 recent_requests 是最新任务在前；超过 120 条时也必须保留最新项。
        requests = [
          {
            "request_id": f"req-{seq}", "sequence": seq,
            "device_id": "client_pi-01", "arrived_at": now - seq,
            "status": "completed", "total_ms": 100,
          }
          for seq in range(150)
        ]
        source = {
          "summary": {"total_requests": 150, "avg_latency_ms": 100, "connected_devices": 1},
          "server": {}, "resources": {}, "dao": {}, "devices": [],
          "recent_requests": requests,
        }
        adapted = dashboard._adapt_admin_snapshot(source)
        event_ids = {event["request_id"] for event in adapted["events"]}
        self.assertIn("req-0", event_ids)
        self.assertNotIn("req-149", event_ids)

    def test_8008_admin_redirects_to_single_8009_page(self):
      with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        dashboard = load_dashboard(Path(temp_dir))
        with TestClient(dashboard.app, follow_redirects=False) as client:
          response = client.get("/admin")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "http://127.0.0.1:8009/admin")

    def test_8009_fallback_pipeline_uses_plain_chinese(self):
      sys.path.insert(0, str(EDUCATION_DIR))
      import web_api

      pipeline = web_api._pipeline_from_dao({"strategy": "offload", "stages_to_offload": []})
      labels = [stage["model"] for stage in pipeline[:4]]
      self.assertEqual(labels, ["视频采集与传输", "人物检测", "姿态检查", "表情与课堂状态识别"])

    def test_browser_camera_frame_decode_rejects_invalid_input(self):
      sys.path.insert(0, str(EDUCATION_DIR))
      import web_api

      with self.assertRaisesRegex(ValueError, "空帧"):
        web_api.decode_browser_camera_frame(b"")
      with self.assertRaisesRegex(ValueError, "无法解码"):
        web_api.decode_browser_camera_frame(b"not-a-jpeg")

    def test_browser_camera_websocket_accepts_binary_jpeg(self):
      sys.path.insert(0, str(EDUCATION_DIR))
      import web_api

      frame = web_api.np.zeros((48, 64, 3), dtype=web_api.np.uint8)
      ok, encoded = web_api.cv2.imencode(".jpg", frame)
      self.assertTrue(ok)
      analysis = {
        "elapsed_ms": 12.5,
        "report": {"state_counts": {"focused": 1}},
        "edge": {"connected": True},
        "image": "data:image/jpeg;base64,test",
      }
      class FakeWebSocket:
        def __init__(self):
          self.sent = []

        async def accept(self):
          return None

        async def receive_json(self):
          return {
            "type": "start",
            "source": "browser_camera",
            "device_id": "browser-test",
            "analysis_mode": "standard",
            "max_frames": 1,
          }

        async def receive(self):
          return {"type": "websocket.receive", "bytes": encoded.tobytes()}

        async def send_text(self, value):
          self.sent.append(json.loads(value))

      socket = FakeWebSocket()
      async def run_inline(function, *args):
        return function(*args)

      with patch.object(web_api, "analyze_classroom_frame", return_value=analysis) as analyze:
        with patch.object(web_api.asyncio, "to_thread", new=run_inline):
          asyncio.run(web_api.classroom_stream_ws(socket))
      self.assertEqual(socket.sent[0]["type"], "started")
      self.assertEqual(socket.sent[1]["type"], "analysis")
      self.assertEqual(socket.sent[1]["frame_index"], 1)
      self.assertEqual(socket.sent[2]["type"], "finished")
      analyze.assert_called_once()
      analyzed_frame = analyze.call_args.args[0]
      self.assertEqual(analyzed_frame.shape[:2], (48, 64))

    def test_classroom_reports_are_kept_per_browser_device(self):
      sys.path.insert(0, str(EDUCATION_DIR))
      import web_api

      payload = {
        "elapsed_ms": 10,
        "report": {"state_counts": {"focused": 1}},
        "edge": {"connected": True},
      }
      with web_api._classroom_report_lock:
        web_api._classroom_latest_reports.clear()
      with patch.object(web_api, "frame_to_thumbnail_data_url", return_value="thumbnail"):
        web_api._store_classroom_report(payload, None, "client_web-one", "A1", "学生一")
        web_api._store_classroom_report(payload, None, "client_web-two", "A2", "学生二")
      with web_api._classroom_report_lock:
        reports = dict(web_api._classroom_latest_reports)
        web_api._classroom_latest_reports.clear()
      self.assertEqual(set(reports), {"client_web-one", "client_web-two"})
      self.assertEqual(reports["client_web-one"]["student_name"], "学生一")
      self.assertEqual(reports["client_web-two"]["seat_label"], "A2")


if __name__ == "__main__":
    unittest.main()
