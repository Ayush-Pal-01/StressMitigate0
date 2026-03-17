"""
main.py — FastAPI entry point for StressMitigate.

Uses @app.lifespan for async model loading.
Registers all routers under /api/v1/*.
Serves the frontend from /static.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.config import CORS_ORIGINS
from backend.database import connect_db, close_db
from backend.ml_service import ml_service

# Import routers
from backend.routers.auth_router import router as auth_router
from backend.routers.user_router import router as user_router
from backend.routers.checkin_router import router as checkin_router
from backend.routers.chat_router import router as chat_router
from backend.routers.wellness_router import router as wellness_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to MongoDB, load all three ML models.
    Shutdown: close DB, release model memory.
    """
    print("=" * 60)
    print("   StressMitigate — Starting Up")
    print("=" * 60)



    # 1. Database
    await connect_db()

    # 2. ML Models (loaded synchronously but only once at startup)
    print("\n🔄 Loading ML models...")

    print("  [1/3] Face Stress Model (MobileNetV2, 64×64 RGB)...")
    ml_service.load_face_model()

    print("  [2/3] Voice Stress Model (Wav2Vec2 TFLite, raw waveform)...")
    ml_service.load_voice_model()

    print("  [3/3] Text Stress Model (BERT + Tokenizer)...")
    ml_service.load_text_model()

    print("\n" + "=" * 60)
    print("   ✅ All systems GO. Server ready.")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down StressMitigate...")
    await close_db()
    print("👋 Goodbye.\n")


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
    return {
        "status": "healthy",
        "models": {
            "face": ml_service.fer_model is not None,
            "voice": ml_service.voice_interpreter is not None,
            "text": ml_service.text_model is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
