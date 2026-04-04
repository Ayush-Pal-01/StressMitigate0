"""
conftest.py — Shared pytest fixtures for StressMitigate tests.

Sets up a test FastAPI client using httpx for async route testing.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.users = MagicMock()
    db.checkins = MagicMock()
    db.conversations = MagicMock()
    return db


@pytest_asyncio.fixture
async def client(mock_db):
    """
    Create an async test client with mocked database.
    Patches get_db to return our mock DB so tests don't need real MongoDB.
    """
    with patch("backend.database._connected", True), \
         patch("backend.database.db", mock_db), \
         patch("backend.database.get_db", return_value=mock_db):

        # Import app AFTER patching to avoid DB connection attempts
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
