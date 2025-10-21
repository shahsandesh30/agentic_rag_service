import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .metrics import HTTP_LATENCY, HTTP_REQUESTS

_log = logging.getLogger("app")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exclude_paths=("/metrics",)):
        super().__init__(app)
        self.exclude_paths = set(exclude_paths)

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.scope.get("path", "")
        method = request.method
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()

        # add request-id header downstream
        request.scope["x-request-id"] = req_id
        if path not in self.exclude_paths:
            _log.info(f"[{req_id}] -> {method} {path}")

        try:
            response: Response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            if path not in self.exclude_paths:
                HTTP_LATENCY.labels(path=path, method=method).observe(duration)
                HTTP_REQUESTS.labels(path=path, method=method, status="500").inc()
                _log.exception(f"[{req_id}] !! {method} {path} failed in {duration:.3f}s")
            raise

        duration = time.perf_counter() - start
        if path not in self.exclude_paths:
            HTTP_LATENCY.labels(path=path, method=method).observe(duration)
            HTTP_REQUESTS.labels(path=path, method=method, status=str(response.status_code)).inc()
            _log.info(f"[{req_id}] <- {method} {path} {response.status_code} in {duration:.3f}s")

        # echo request id for clients
        response.headers["x-request-id"] = req_id
        return response
