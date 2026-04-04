"""
test_auth.py — Tests for authentication endpoints.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId


@pytest.mark.asyncio
async def test_register_success(client, mock_db):
    """Test successful user registration."""
    mock_db.users.find_one = AsyncMock(return_value=None)
    mock_db.users.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=ObjectId())
    )

    response = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "securepass123",
        "display_name": "Test User"
    })

    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["display_name"] == "Test User"


@pytest.mark.asyncio
async def test_register_duplicate_email(client, mock_db):
    """Test registration with already existing email."""
    mock_db.users.find_one = AsyncMock(return_value={"email": "test@example.com"})

    response = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "securepass123",
    })

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_register_short_password(client, mock_db):
    """Test registration with password shorter than 6 characters."""
    response = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "12345",
    })

    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_register_invalid_email(client, mock_db):
    """Test registration with invalid email format."""
    response = await client.post("/api/v1/auth/register", json={
        "email": "not-an-email",
        "password": "securepass123",
    })

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client, mock_db):
    """Test successful login."""
    from backend.auth import hash_password

    hashed_pw = hash_password("securepass123")
    user_doc = {
        "_id": ObjectId(),
        "email": "test@example.com",
        "display_name": "Test User",
        "hashed_password": hashed_pw,
    }
    mock_db.users.find_one = AsyncMock(return_value=user_doc)

    response = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "securepass123",
    })

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["display_name"] == "Test User"


@pytest.mark.asyncio
async def test_login_wrong_password(client, mock_db):
    """Test login with incorrect password."""
    from backend.auth import hash_password

    user_doc = {
        "_id": ObjectId(),
        "email": "test@example.com",
        "hashed_password": hash_password("correctpassword"),
    }
    mock_db.users.find_one = AsyncMock(return_value=user_doc)

    response = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "wrongpassword",
    })

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client, mock_db):
    """Test login with email that doesn't exist."""
    mock_db.users.find_one = AsyncMock(return_value=None)

    response = await client.post("/api/v1/auth/login", json={
        "email": "ghost@example.com",
        "password": "anypassword",
    })

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_email_case_normalization(client, mock_db):
    """Test that emails are normalized to lowercase."""
    mock_db.users.find_one = AsyncMock(return_value=None)
    mock_db.users.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=ObjectId())
    )

    response = await client.post("/api/v1/auth/register", json={
        "email": "Test@EXAMPLE.com",
        "password": "securepass123",
    })

    assert response.status_code == 201
    # Verify the email was normalized when saving
    call_args = mock_db.users.insert_one.call_args[0][0]
    assert call_args["email"] == "test@example.com"
