import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("catalyx")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        request.state.request_id = request_id

        logger.info(
            f"request_id={request_id} method={request.method} "
            f"path={request.url.path} query={dict(request.query_params)}"
        )

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"request_id={request_id} status={response.status_code} "
            f"duration_ms={duration_ms}"
        )

        response.headers["X-Request-ID"] = request_id
        return response