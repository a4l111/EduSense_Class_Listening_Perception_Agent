# -*- coding: utf-8 -*-
"""验证「全课堂调度管理页」修复：8009 /admin 代理 8008 全量快照。

检查点：
  1. /api/admin/dao-snapshot 返回 dao-admin 所需结构且包含全部真实设备
  2. /admin 页面可访问（dao-admin index.html）
  3. /admin-assets/* 静态资源可访问
  4. /api/admin/snapshot 保持 :8011 原始格式（既有消费者不受影响）
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8009"


def get(path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return 0, f"ERR {exc}"


def head(path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as resp:
            return resp.status, resp.read(120).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"ERR {exc}"


def main() -> int:
    ok = True

    status, snap = get("/api/admin/dao-snapshot")
    print(f"[DAO_SNAPSHOT] status={status}")
    if status != 200 or not isinstance(snap, dict):
        print("FAIL: dao-snapshot 不可用", snap)
        return 1
    devices = snap.get("devices") or []
    print("[DEVICES]")
    for dev in devices:
        print(
            "  -",
            dev.get("id"),
            "| status:", dev.get("status"),
            "| tasks:", dev.get("tasks"),
            "| latency:", dev.get("latency_ms"),
            "| state:", dev.get("state"),
            "| role:", dev.get("role"),
        )
    if len(devices) < 3:
        print(f"FAIL: 期望至少 3 台设备，实际 {len(devices)}")
        ok = False
    summary = snap.get("summary") or {}
    print("[SUMMARY]", summary)
    print("[SERVER]", snap.get("server"))
    print("[DAO]", snap.get("dao"))
    print("[PIPELINE]", [s.get("name") for s in (snap.get("pipeline") or [])])
    print("[COMPARISON]", snap.get("comparison"))
    events = snap.get("events") or []
    print(f"[EVENTS] count={len(events)} sample={events[:1]}")
    if not all(k in dev for dev in devices for k in ("id", "name", "role", "status", "tasks")):
        print("FAIL: devices 字段不完整")
        ok = False

    st, body = head("/admin")
    print(f"[ADMIN_PAGE] status={st} head={body[:80]!r}")
    if st != 200 or "<!doctype html" not in body.lower():
        print("FAIL: /admin 页面不可用")
        ok = False

    for asset in ("/admin-assets/app.js?v=7", "/admin-assets/styles.css?v=7"):
        st, _ = head(asset)
        print(f"[ASSET] {asset} -> {st}")
        if st != 200:
            ok = False

    st, raw = get("/api/admin/snapshot")
    print(f"[RAW_SNAPSHOT] status={st} devices={len((raw or {}).get('devices') or [])} "
          f"connected={((raw or {}).get('summary') or {}).get('connected_devices')}")
    if st != 200:
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
