"""Dashboard module for ActiveContext monitoring.

Provides a web-based dashboard for real-time monitoring of ActiveContext sessions.

Usage:
    The dashboard is started via the /dashboard slash command in the ACP transport.

    /dashboard start [port]  - Start dashboard server (default port: 8765)
    /dashboard stop          - Stop dashboard server
    /dashboard status        - Show dashboard status
    /dashboard [open]        - Open dashboard in browser (auto-starts if needed)
"""

from activecontext.dashboard.routes import broadcast_update
from activecontext.dashboard.server import (
    get_dashboard_status,
    is_dashboard_running,
    start_dashboard,
    stop_dashboard,
)

__all__ = [
    "start_dashboard",
    "stop_dashboard",
    "is_dashboard_running",
    "get_dashboard_status",
    "broadcast_update",
]
