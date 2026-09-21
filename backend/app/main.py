from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.generate import (
    router as generate_router,
)

from backend.app.api.health import (
    router as health_router,
)

from backend.app.api.download import (
    router as download_router,
)


app = FastAPI(
    title="Paper Agent API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^https?://"
        r"(localhost|127\.0\.0\.1)"
        r"(:\d+)?$"
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    health_router
)

app.include_router(
    generate_router
)

app.include_router(
    download_router
)