"""
auth_router.py — Registration, Login, Profile Update & Password Change endpoints.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from bson import ObjectId
from backend.schemas import RegisterRequest, LoginRequest, TokenResponse
from backend.auth import hash_password, verify_password, create_token, get_current_user
from backend.database import get_db
from pydantic import BaseModel, EmailStr
from typing import Optional

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ── Additional request models ──
class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """
    Register a new user with email + password.
    Automatically logs them in by returning a JWT.
    """
    db = get_db()
    normalized_email = req.email.strip().lower()

    # Check if email already exists (case-insensitive)
    existing = await db.users.find_one({"email": normalized_email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Create user document
    user_doc = {
        "email": normalized_email,
        "display_name": req.display_name or "User",
        "hashed_password": hash_password(req.password),
        "communication_style": "Gentle & Reassuring",
        "privacy_settings": {
            "allow_text": True,
            "allow_voice": True,
            "allow_camera": False,
        },
        "created_at": datetime.now(timezone.utc),
    }
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)

    # Return JWT immediately — smart login after register
    token = create_token(user_id, normalized_email)
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        display_name=user_doc["display_name"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """
    Authenticate with email + password, return a JWT.
    """
    db = get_db()
    normalized_email = req.email.strip().lower()
    user = await db.users.find_one({"email": normalized_email})

    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    user_id = str(user["_id"])
    token = create_token(user_id, normalized_email)
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        display_name=user.get("display_name", "User"),
    )


@router.put("/profile")
async def update_profile(req: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    """
    Update user's display name and/or email.
    """
    db = get_db()
    updates = {}

    if req.display_name and req.display_name.strip():
        updates["display_name"] = req.display_name.strip()

    if req.email:
        new_email = req.email.strip().lower()
        # Check if new email is taken by someone else
        existing = await db.users.find_one({"email": new_email})
        if existing and str(existing["_id"]) != str(user["_id"]):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This email is already in use.",
            )
        updates["email"] = new_email

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    await db.users.update_one({"_id": user["_id"]}, {"$set": updates})
    return {"message": "Profile updated successfully.", "updated": list(updates.keys())}


@router.put("/change-password")
async def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """
    Change password. Requires current password for verification.
    """
    db = get_db()

    # Verify current password
    if not verify_password(req.current_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    if len(req.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters.",
        )

    # Hash and save new password
    new_hash = hash_password(req.new_password)
    await db.users.update_one({"_id": user["_id"]}, {"$set": {"hashed_password": new_hash}})
    return {"message": "Password changed successfully."}

