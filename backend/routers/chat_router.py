"""
chat_router.py — The main multimodal endpoint.

Accepts: multipart/form-data with optional text, audio, image, session_id.
Returns: detected emotions, AI response, modalities used, timestamp.

Handles ALL 7 hybrid modality combinations + 1 empty case (8 total).
Privacy-first: respects user's privacy_settings toggles.

UPDATED: Fetches recent chat history for context windowing and
classifies text stress into 3 tiers (No/Low/High).
"""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from datetime import datetime, timezone
import asyncio
import uuid

from backend.schemas import ChatResponse, DetectedEmotions
from backend.auth import get_current_user
from backend.ml_service import ml_service
from backend.brain import generate_response
from backend.database import get_db
from backend.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat (Multimodal)"])

# ─── Upload validation constants ───
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/x-wav"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024   # 5 MB


def _classify_text_stress(result: dict) -> dict:
    """
    Convert BERT's binary (No Stress / STRESS DETECTED) into 3-tier classification
    using confidence thresholds.

    Logic:
      - "STRESS DETECTED" with confidence > 97.5% → "High Stress"
      - "STRESS DETECTED" with confidence 84-97.5% → "Low Stress"
      - "STRESS DETECTED" with confidence < 84% → "Low Stress"
      - "No Stress" with confidence > 84% → "No Stress"
      - "No Stress" with confidence <= 84% → "Low Stress"
    """
    label = result.get("label", "")
    conf = result.get("confidence", 0.0)

    if label == "STRESS DETECTED":
        if conf > 0.975:
            return {"label": "High Stress", "confidence": conf}
        else:
            return {"label": "Low Stress", "confidence": conf}
    elif label == "No Stress":
        if conf > 0.84:
            return {"label": "No Stress", "confidence": conf}
        else:
            return {"label": "Low Stress", "confidence": conf}
    return result


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    text: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
    session_id: str = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Main multimodal chat endpoint.

    Hybrid Model Cases Handled (8 total):
    ──────────────────────────────────────
    Case 1: Text only
    Case 2: Voice only
    Case 3: Face only
    Case 4: Text + Voice
    Case 5: Text + Face
    Case 6: Voice + Face
    Case 7: Text + Voice + Face (full multimodal)
    Case 8: No input provided (returns guidance message)

    Privacy: If user has disabled a modality in settings, it is silently
    skipped even if data is sent.
    """
    db = get_db()
    privacy = user.get("privacy_settings", {})
    style = user.get("communication_style", "Gentle & Reassuring")
    emotions = {}
    modalities_used = []
    user_id = str(user["_id"])

    # Use provided session_id or generate new one
    if not session_id:
        session_id = str(uuid.uuid4())

    # ─── Text Analysis ───
    # Text is ALWAYS checked first (mandatory when provided)
    if text and text.strip() and privacy.get("allow_text", True):
        result = await asyncio.to_thread(ml_service.predict_text, text)
        if result["label"] != "unavailable":
            # Apply 3-tier classification
            result = _classify_text_stress(result)
            emotions["text"] = result
            modalities_used.append("text")

    # ─── Voice Analysis ───
    if audio and privacy.get("allow_voice", True):
        # Validate MIME type
        if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported audio format: {audio.content_type}. Accepted: {', '.join(ALLOWED_AUDIO_TYPES)}",
            )
        audio_bytes = await audio.read()
        # Validate size
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large ({len(audio_bytes) // (1024*1024)}MB). Maximum: 10MB.",
            )
        if len(audio_bytes) > 0:
            result = await asyncio.to_thread(ml_service.predict_voice, audio_bytes)
            if result["label"] not in ("unavailable", "error", "silent_audio"):
                emotions["voice"] = result
                modalities_used.append("voice")
            elif result["label"] == "silent_audio":
                # Still include it with a note, but don't count as strong signal
                emotions["voice"] = result

    # ─── Face Analysis ───
    if image and privacy.get("allow_camera", False):
        # Validate MIME type
        if image.content_type and image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format: {image.content_type}. Accepted: {', '.join(ALLOWED_IMAGE_TYPES)}",
            )
        image_bytes = await image.read()
        # Validate size
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Image file too large ({len(image_bytes) // (1024*1024)}MB). Maximum: 5MB.",
            )
        if len(image_bytes) > 0:
            result = await asyncio.to_thread(ml_service.predict_face, image_bytes)
            if result["label"] not in ("unavailable", "error", "no_face_detected"):
                emotions["face"] = result
                modalities_used.append("face")
            elif result["label"] == "no_face_detected":
                emotions["face"] = result

    # ─── Case 8: No input at all ───
    if not emotions and not text:
        return ChatResponse(
            detected_emotions=DetectedEmotions(),
            ai_response="I'm here whenever you're ready. You can type a message, send a voice note, or turn on your camera — whatever feels comfortable.",
            modalities_used=[],
            timestamp=datetime.now(timezone.utc),
        )

    # ─── Fetch recent chat history for context windowing ───
    chat_history = []
    try:
        cursor = db.conversations.find(
            {"user_id": user_id, "session_id": session_id},
            {"message_payload": 1, "ai_response": 1, "_id": 0},
        ).sort("timestamp", -1).limit(10)

        recent_messages = await cursor.to_list(length=10)
        # Reverse to get chronological order
        for msg in reversed(recent_messages):
            if msg.get("message_payload"):
                chat_history.append({"role": "user", "text": msg["message_payload"]})
            if msg.get("ai_response"):
                chat_history.append({"role": "model", "text": msg["ai_response"]})
    except Exception as e:
        logger.warning(f"Could not fetch chat history: {e}")

    # ─── Brain Pipeline: Fusion → Gemini (with context) ───
    ai_response = await generate_response(
        emotions, text, style, chat_history=chat_history
    )

    # ─── Persist to MongoDB ───
    conversation_doc = {
        "user_id": user_id,
        "session_id": session_id,
        "message_payload": text,
        "detected_emotions": {k: v for k, v in emotions.items()},
        "modalities_used": modalities_used,
        "ai_response": ai_response,
        "timestamp": datetime.now(timezone.utc),
    }
    await db.conversations.insert_one(conversation_doc)

    # ─── Build confidence dict ───
    confidence_dict = {}
    for mod, res in emotions.items():
        if isinstance(res, dict) and "confidence" in res:
            confidence_dict[mod] = res["confidence"]

    return ChatResponse(
        detected_emotions=DetectedEmotions(
            text=emotions.get("text", {}).get("label"),
            voice=emotions.get("voice", {}).get("label"),
            face=emotions.get("face", {}).get("label"),
            confidence=confidence_dict if confidence_dict else None,
        ),
        ai_response=ai_response,
        modalities_used=modalities_used,
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
    )
