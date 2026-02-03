from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.api.models import ErrorOutput
from src.exceptions import CourseMLOpsError
from src.exceptions import InvalidInputError
from src.exceptions import ModelNotLoadedError


def course_mlops_exception_handler(
    request: Request,
    exc: CourseMLOpsError,
) -> JSONResponse:
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, ModelNotLoadedError):
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, InvalidInputError):
        http_status = status.HTTP_400_BAD_REQUEST

    return JSONResponse(
        status_code=http_status,
        content=ErrorOutput(
            detail=exc.message,
            error_code=exc.error_code,
        ).model_dump(),
    )


def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Internal server error"

    if isinstance(exc, HTTPException):
        http_status = exc.status_code
        detail = exc.detail
        if http_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            print(f"[-] {detail}")
        else:
            print(f"[?] {detail}")

    elif isinstance(exc, RequestValidationError):
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
        messages = [str(e.get("msg", "")) for e in exc.errors()]
        detail = "; ".join(messages) if messages else "Validation error"
        print(f"[-] Validation error on {request.method} {request.url}: {detail}")

    else:
        print(f"[-] Unhandled exception on {request.method} {request.url}")

    return JSONResponse(
        status_code=http_status,
        content=ErrorOutput(detail=detail).model_dump(),
    )
