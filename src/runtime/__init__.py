"""运行模式相关组件。"""

from __future__ import annotations

__all__ = ["RuntimeAnnouncer"]


def __getattr__(name: str):
    """按需导入，避免 monitor_server 启动时强依赖 zeroconf。"""
    if name == "RuntimeAnnouncer":
        from .discovery import RuntimeAnnouncer

        return RuntimeAnnouncer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
