from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging

from schemas import ErrorResponse
from model_storage import load_churn_model

from api.dataset import router as dataset_router
from api.predictions import router as predictions_router
from api.model import router as model_router
from api.health import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_data = load_churn_model()
    except FileNotFoundError:
        app.state.model_data = None

    yield

app = FastAPI(lifespan=lifespan)

app.include_router(dataset_router)
app.include_router(predictions_router)
app.include_router(model_router)
app.include_router(health_router)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    log_method = (
        logger.error
        if exc.status_code >= 500
        else logger.warning
    )

    log_method(
        "HTTP-ошибка: method=%s path=%s status=%s message=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    error = ErrorResponse(
        code=exc.status_code,
        message=str(exc.detail),
        details=None,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(mode="json"),
        headers=exc.headers,
    )

@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Необработанная ошибка: %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    error = ErrorResponse(
        code=500,
        message="Внутренняя ошибка сервиса. Попробуйте повторить запрос позже.",
        details=None,
    )

    return JSONResponse(
        status_code=500,
        content=error.model_dump(mode="json"),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = []

    for item in exc.errors():
        errors.append(
            {
                "location": list(item["loc"]),
                "message": item["msg"],
                "type": item["type"],
            }
        )

    logger.warning(
        "Ошибка валидации: method=%s path=%s errors=%s",
        request.method,
        request.url.path,
        errors,
    )

    error = ErrorResponse(
        code=422,
        message="Некорректные данные запроса.",
        details={"errors": errors},
    )

    return JSONResponse(
        status_code=422,
        content=error.model_dump(mode="json"),
    )