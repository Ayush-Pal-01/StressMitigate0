"""
user_router.py — User preferences endpoints (privacy toggles & persona).
"""
from fastapi import APIRouter, Depends
from backend.schemas import UserPreferences
from backend.auth import get_current_user
from backend.database import get_db

router = APIRouter(prefix="/api/v1/user", tags=["User Preferences"])


@router.get("/preferences", response_model=UserPreferences)
async def get_preferences(user: dict = Depends(get_current_user)):
    """Return the current user's communication style and privacy settings."""
    return UserPreferences(
        communication_style=user.get("communication_style", "Gentle & Reassuring"),
        privacy_settings=user.get("privacy_settings", {}),
    )


@router.put("/preferences", response_model=UserPreferences)
async def update_preferences(prefs: UserPreferences, user: dict = Depends(get_current_user)):
    """Update the user's communication style and privacy toggles."""
    db = get_db()
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "communication_style": prefs.communication_style,
            "privacy_settings": prefs.privacy_settings.model_dump(),
        }},
    )
    return prefs
