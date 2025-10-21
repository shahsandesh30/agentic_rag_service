# app/obs/request_log.py
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("http")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """
    Lightweight request logging.

    - Adds/propagates X-Request-ID
    - Logs: method, path, status, duration (ms), request_id
    - Skips very noisy paths (healthz, metrics) by default
    """

    def __init__(
        self,
        app,
        *,
        skip_paths: Iterable[str] = ("/healthz", "/metrics"),
        header_name: str = "X-Request-ID",
    ):
        super().__init__(app)
        self.skip_paths = set(skip_paths)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        # Skip noisy endpoints
        if request.url.path in self.skip_paths:
            return await call_next(request)

        started = time.perf_counter()

        # Get or create request id
        req_id = request.headers.get(self.header_name)
        if not req_id:
            req_id = str(uuid.uuid4())

        # Process the request
        try:
            response: Response = await call_next(request)
        except Exception:
            # Ensure we still log duration and request id on exceptions
            duration_ms = (time.perf_counter() - started) * 1000.0
            log.exception(
                "request method=%s path=%s status=%s dur_ms=%.1f request_id=%s",
                request.method,
                request.url.path,
                500,
                duration_ms,
                req_id,
            )
            raise

        # Add request id to response
        response.headers.setdefault(self.header_name, req_id)

        # Duration and final log line
        duration_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "request method=%s path=%s status=%s dur_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            req_id,
        )
        return response
