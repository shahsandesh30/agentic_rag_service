# app/http_errors.py
import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


def _base_payload(error: str, message: str, request: Request, detail: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": error,
        "message": message,
        "path": request.url.path,
        "method": request.method,
    }

    if detail is not None:
        payload["detail"] = detail

    return payload


async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Raised when Pydantic/FastAPI request body/query/path validation fails.
    """
    log.info("422 validation_error at %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content=_base_payload(
            error="validation_error",
            message="The request failed validation.",
            request=request,
            detail=exc.errors(),  # list of {'loc':..., 'msg':..., 'type':...}
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handles explicit HTTPException (404, 401, 400, etc.) raised by routes or dependencies.
    """
    # Avoid double-logging 404 noise at ERROR level
    if exc.status_code >= 500:
        log.exception(
            "HTTP %s at %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail
        )
    else:
        log.warning(
            "HTTP %s at %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail
        )

    # exc.detail may be str or dict; we normalize to "message" + optional "detail"
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    detail = None if isinstance(exc.detail, str) else exc.detail

    return JSONResponse(
        status_code=exc.status_code,
        content=_base_payload(
            error="http_error",
            message=str(message),
            request=request,
            detail=detail,
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler for any unhandled exceptions. Returns a 500 without leaking internals.
    """
    log.exception("Unhandled error at %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_base_payload(
            error="internal_server_error",
            message="An unexpected error occurred.",
            request=request,
            # Do NOT include internal stack traces in the response.
        ),
    )
