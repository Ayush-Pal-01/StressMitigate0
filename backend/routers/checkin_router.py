"""
checkin_router.py — Discrete mood check-in endpoint with BERT stress analysis.

Now runs BERT text analysis on optional_notes to detect stress category and score.
Saves stress_category and stress_score alongside the mood check-in.
"""
from fastapi import APIRouter, Depends
from datetime import datetime, timezone
import asyncio
from backend.schemas import CheckInRequest
from backend.auth import get_current_user
from backend.database import get_db
from backend.ml_service import ml_service

router = APIRouter(prefix="/api/v1/analyze", tags=["Check-In"])

# Mood → calming suggestion mapping
SUGGESTIONS = {
    "Calm": "Beautiful! Keep this peaceful momentum going. Maybe try a 2-minute gratitude reflection?",
    "Neutral": "A steady state is a great place to be. How about a short mindfulness exercise to stay centered?",
    "Anxious": "I hear you. Let's take a slow, deep breath together — inhale for 4 counts, hold for 4, exhale for 6.",
    "Stressed": "Thank you for sharing. You deserve a moment of calm. Ready for a guided breathing exercise?",
}


def _classify_stress(result: dict) -> dict:
    """Convert BERT binary output to 3-tier classification."""
    label = result.get("label", "")
    conf = result.get("confidence", 0.0)

    if label == "STRESS DETECTED":
        if conf > 0.975:
            return {"category": "High Stress", "score": round(conf * 100, 1)}
        else:
            return {"category": "Low Stress", "score": round(conf * 100, 1)}
    elif label == "No Stress":
        if conf > 0.84:
            return {"category": "No Stress", "score": round((1 - conf) * 100, 1)}
        else:
            return {"category": "Low Stress", "score": round((1 - conf) * 100, 1)}
    return {"category": "Unknown", "score": 50.0}


@router.post("/check-in")
async def check_in(data: CheckInRequest, user: dict = Depends(get_current_user)):
    """
    Save a discrete mood state, optionally analyze text for stress,
    and return an immediate calming suggestion with stress analysis results.
    """
    db = get_db()
    now = datetime.now(timezone.utc)

    doc = {
        "user_id": str(user["_id"]),
        "mood_state": data.mood_state,
        "optional_notes": data.optional_notes,
        "timestamp": now,
    }

    # Run BERT text analysis on notes if provided
    stress_analysis = None
    if data.optional_notes and data.optional_notes.strip():
        bert_result = await asyncio.to_thread(ml_service.predict_text, data.optional_notes)
        if bert_result.get("label") != "unavailable":
            stress_analysis = _classify_stress(bert_result)
            doc["stress_category"] = stress_analysis["category"]
            doc["stress_score"] = stress_analysis["score"]
            doc["bert_raw"] = bert_result

    await db.checkins.insert_one(doc)

    suggestion = SUGGESTIONS.get(data.mood_state, SUGGESTIONS["Neutral"])

    response = {
        "message": f"Mood '{data.mood_state}' logged successfully.",
        "suggestion": suggestion,
        "timestamp": now.isoformat(),
    }

    if stress_analysis:
        response["stress_category"] = stress_analysis["category"]
        response["stress_score"] = stress_analysis["score"]

    return response
