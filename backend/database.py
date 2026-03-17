"""
database.py — Async MongoDB connection using Motor.
Collections: users, checkins, conversations.

Non-blocking: if MongoDB is not running, the server starts anyway
and logs a warning. Endpoints that need the DB will return 503.
"""
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException, status
from backend.config import MONGO_URI, MONGO_DB_NAME

client: AsyncIOMotorClient = None
db = None
_connected = False


async def connect_db():
    """Open MongoDB connection and return the database handle."""
    global client, db, _connected
    try:
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a quick connection check
        await client.server_info()
        db = client[MONGO_DB_NAME]

        # Ensure indexes for fast lookups
        await db.users.create_index("email", unique=True)
        await db.checkins.create_index("user_id")
        await db.conversations.create_index([("user_id", 1), ("session_id", 1)])

        _connected = True
        print("✅ MongoDB connected & indexes ensured.")
        return db
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("   The server will start, but auth/data endpoints will return 503.")
        print("   Start MongoDB and restart the server when ready.")
        _connected = False
        return None


async def close_db():
    """Gracefully close the MongoDB connection."""
    global client
    if client:
        client.close()
        print("🛑 MongoDB connection closed.")


def get_db():
    """Return the current database handle. Raises 503 if not connected."""
    if not _connected or db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not available. Please ensure MongoDB is running.",
        )
    return db
