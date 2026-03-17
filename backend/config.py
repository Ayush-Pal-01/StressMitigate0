"""
config.py — Central configuration for StressMitigate backend.
All environment variables, paths, and constants in one place.
Loads from .env file automatically via python-dotenv.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------- Base Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "Models"

# ---------- Model Paths ----------
# FSM: MobileNetV2, 64×64 RGB, 3 classes (High Stress, Low Stress, No Stress)
FSM_MODEL_PATH = str(MODELS_DIR / "FSM" / "fine_tuned_stress_model.keras")
FSM_IMG_SIZE = (64, 64)
FSM_CLASSES = ["High Stress", "Low Stress", "No Stress"]

# VSM: Wav2Vec2 → TFLite, raw waveform 80 000 samples, 3 classes
VSM_MODEL_PATH = str(MODELS_DIR / "VSM" / "stress_model_ultra.tflite")
VSM_SAMPLE_RATE = 16_000
VSM_MAX_DURATION = 5  # seconds
VSM_TARGET_LEN = VSM_SAMPLE_RATE * VSM_MAX_DURATION  # 80 000
VSM_CLASSES = ["No Stress", "Low Stress", "High Stress"]

# Text: BertForSequenceClassification, tokenizer + tf_model.h5, 2 classes
TEXT_MODEL_DIR = str(MODELS_DIR / "saved_stress_model")
TEXT_CLASSES = ["No Stress", "STRESS DETECTED"]
TEXT_MAX_LEN = 128

# ---------- MongoDB ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stress_mitigate")

# ---------- JWT Auth ----------
JWT_SECRET = os.getenv("JWT_SECRET", "StressMitigate_SuperSecret_2026_ChangeInProd!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ---------- Gemini ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_FALLBACK_MODEL = "gemini-1.5-flash"

# ---------- CORS ----------
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5500", "*"]

# ---------- Fallback Response ----------
LLM_FALLBACK = (
    "I'm here for you. Let's take a deep breath together. "
    "My connection is a bit slow right now, but I am listening."
)
