"""
main.py — FastAPI entry point for StressMitigate.

Uses @app.lifespan for async model loading.
Registers all routers under /api/v1/*.
Serves the frontend from /static.
"""
import os
import sys
import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.config import CORS_ORIGINS
from backend.database import connect_db, close_db
import backend.database as _db_module
from backend.ml_service import ml_service
from backend.logger import get_logger

# Import routers
from backend.routers.auth_router import router as auth_router
from backend.routers.user_router import router as user_router
from backend.routers.checkin_router import router as checkin_router
from backend.routers.chat_router import router as chat_router
from backend.routers.wellness_router import router as wellness_router

logger = get_logger(__name__)

# Optional: Sentry for error tracking
try:
    import sentry_sdk
    SENTRY_DSN = os.getenv("SENTRY_DSN", "")
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.2,
            profiles_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        logger.info(f"Sentry initialized (environment: {os.getenv('ENVIRONMENT', 'development')})")
except ImportError:
    pass  # sentry-sdk not installed — that's fine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to MongoDB, load all three ML models.
    Shutdown: close DB, release model memory.
    """
    logger.info("=" * 50)
    logger.info("StressMitigate — Starting Up")
    logger.info("=" * 50)

    # 1. Database
    await connect_db()

    # 2. ML Models (loaded synchronously but only once at startup)
    logger.info("Loading ML models...")

    logger.info("[1/3] Face Stress Model (MobileNetV2, 64×64 RGB)...")
    ml_service.load_face_model()

    logger.info("[2/3] Voice Stress Model (Wav2Vec2 TFLite, raw waveform)...")
    ml_service.load_voice_model()

    logger.info("[3/3] Text Stress Model (BERT + Tokenizer)...")
    ml_service.load_text_model()

    logger.info("=" * 50)
    logger.info("All systems GO. Server ready.")
    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("Shutting down StressMitigate...")
    await close_db()
    logger.info("Goodbye.")


# ── Create app ──
app = FastAPI(
    title="StressMitigate API",
    description="Privacy-first multimodal AI for emotional well-being.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(checkin_router)
app.include_router(chat_router)
app.include_router(wellness_router)

# ── Serve static frontend files ──
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["Frontend"], response_class=HTMLResponse)
async def root():
    """Serve the main SPA frontend."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health", tags=["Health"])
async def health():
    """
    Enhanced health check — reports model status, database connectivity,
    memory usage, and Python version.
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    # Determine which model backends are loaded
    text_backend = "none"
    if ml_service._use_onnx_text and ml_service.text_onnx_session:
        text_backend = "onnx_int8"
    elif ml_service.text_model:
        text_backend = "tensorflow"

    face_backend = "none"
    if ml_service._use_tflite_face and ml_service.fer_interpreter:
        face_backend = "tflite"
    elif ml_service.fer_model:
        face_backend = "keras"

    voice_backend = "tflite" if ml_service.voice_interpreter else "none"

    return {
        "status": "healthy",
        "models": {
            "text": {
                "loaded": text_backend != "none",
                "backend": text_backend,
            },
            "face": {
                "loaded": face_backend != "none",
                "backend": face_backend,
            },
            "voice": {
                "loaded": voice_backend != "none",
                "backend": voice_backend,
            },
        },
        "database": {
            "connected": _db_module._connected,
        },
        "system": {
            "python_version": sys.version,
            "memory_rss_mb": round(memory_info.rss / (1024 * 1024), 1),
            "memory_vms_mb": round(memory_info.vms / (1024 * 1024), 1),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
