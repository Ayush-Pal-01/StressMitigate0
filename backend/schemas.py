"""
schemas.py — Pydantic models for request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict
from datetime import datetime


# ────────────── Auth ──────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="Min 6 characters")
    display_name: Optional[str] = "User"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    display_name: str


# ────────────── Preferences ──────────────
class PrivacySettings(BaseModel):
    allow_text: bool = True
    allow_voice: bool = True
    allow_camera: bool = False


class UserPreferences(BaseModel):
    communication_style: str = "Gentle & Reassuring"
    privacy_settings: PrivacySettings = PrivacySettings()


# ────────────── Check-In ──────────────
class CheckInRequest(BaseModel):
    mood_state: str = Field(..., description="Calm / Neutral / Anxious / Stressed")
    optional_notes: Optional[str] = None


class CheckInResponse(BaseModel):
    message: str
    suggestion: str
    timestamp: datetime


# ────────────── Chat / Multimodal ──────────────
class DetectedEmotions(BaseModel):
    text: Optional[str] = None
    voice: Optional[str] = None
    face: Optional[str] = None
    confidence: Optional[Dict[str, float]] = None


class ChatResponse(BaseModel):
    detected_emotions: DetectedEmotions
    ai_response: str
    modalities_used: list
    timestamp: datetime
    session_id: Optional[str] = None
