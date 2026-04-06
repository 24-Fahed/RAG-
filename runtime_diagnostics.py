"""Runtime diagnostics helpers for FastAPI services."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request


def _ensure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


def install_runtime_diagnostics(app: FastAPI, service_name: str, mode: str | None = None) -> None:
    """Install request, startup, shutdown, and signal logging on a FastAPI app."""
    if getattr(app.state, "_runtime_diagnostics_installed", False):
        return

    _ensure_logging()
    logger = logging.getLogger(f"{service_name}.runtime")

    active_requests: dict[str, dict] = {}
    active_lock = threading.Lock()
    shutdown_started_at: float | None = None

    def _runtime_context() -> str:
        return (
            f"service={service_name} mode={mode or 'unknown'} "
            f"pid={os.getpid()} ppid={os.getppid()} cwd={os.getcwd()}"
        )

    def _active_request_count() -> int:
        with active_lock:
            return len(active_requests)

    def _log_active_requests(prefix: str) -> None:
        with active_lock:
            current = list(active_requests.values())

        logger.warning("%s active_request_count=%s", prefix, len(current))
        for item in current[:20]:
            elapsed = time.time() - item["started_at"]
            logger.warning(
                "%s request_id=%s method=%s path=%s client=%s elapsed=%.2fs",
                prefix,
                item["request_id"],
                item["method"],
                item["path"],
                item["client"],
                elapsed,
            )

    def _signal_handler(signum: int, _frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning(
            "received_shutdown_signal signal=%s %s thread=%s",
            sig_name,
            _runtime_context(),
            threading.current_thread().name,
        )
        _log_active_requests("shutdown_signal")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            logger.exception("failed_to_register_signal_handler signal=%s", sig)

    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, _signal_handler)
        except Exception:
            logger.exception("failed_to_register_signal_handler signal=SIGHUP")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Callable):
        request_id = uuid.uuid4().hex[:8]
        started_at = time.time()
        client_host = request.client.host if request.client else "unknown"

        item = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": client_host,
            "started_at": started_at,
        }
        with active_lock:
            active_requests[request_id] = item

        logger.info(
            "request_started request_id=%s method=%s path=%s client=%s active_request_count=%s",
            request_id,
            request.method,
            request.url.path,
            client_host,
            _active_request_count(),
        )

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.time() - started_at
            logger.exception(
                "request_failed request_id=%s method=%s path=%s elapsed=%.2fs",
                request_id,
                request.method,
                request.url.path,
                elapsed,
            )
            raise
        finally:
            with active_lock:
                active_requests.pop(request_id, None)

        elapsed = time.time() - started_at
        logger.info(
            "request_finished request_id=%s method=%s path=%s status=%s elapsed=%.2fs active_request_count=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            _active_request_count(),
        )
        return response

    @app.on_event("startup")
    async def _on_startup() -> None:
        logger.info("startup %s", _runtime_context())

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        nonlocal shutdown_started_at
        shutdown_started_at = time.time()
        logger.warning("shutdown_started %s", _runtime_context())
        _log_active_requests("shutdown_started")

    @app.on_event("shutdown")
    async def _on_shutdown_complete_marker() -> None:
        elapsed = 0.0
        if shutdown_started_at is not None:
            elapsed = time.time() - shutdown_started_at
        logger.warning("shutdown_completed %s elapsed=%.2fs", _runtime_context(), elapsed)

    app.state._runtime_diagnostics_installed = True
