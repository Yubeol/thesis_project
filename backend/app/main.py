from fastapi import FastAPI

from backend.app.api.generate import (
    router as generate_router,
)

from backend.app.api.health import (
    router as health_router,
)


app = FastAPI(
    title="Paper Agent API",
    version="1.0.0",
)


app.include_router(
    health_router
)

app.include_router(
    generate_router
)