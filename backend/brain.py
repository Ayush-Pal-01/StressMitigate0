"""
brain.py — The "Brain" Pipeline: Emotion Fusion + Gemini API Integration.

Fuses outputs from Text, Voice, and Face models, constructs a dynamic
system prompt for the Gemini API, handles conflicting emotions, and
provides a safe fallback on LLM failure.

Uses Gemini chat sessions with conversation history for multi-turn,
context-aware interactions.

Retry strategy: Tenacity with exponential backoff + model fallback
for 429 rate-limit errors and transient failures.
"""
import asyncio
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from backend.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL, LLM_FALLBACK
from backend.logger import get_logger

logger = get_logger(__name__)

# Configure Gemini once at module level
_gemini_configured = False
_gemini_model = None


def _ensure_gemini():
    """Configure Gemini SDK once. Returns the model or None."""
    global _gemini_configured, _gemini_model
    if _gemini_configured:
        return _gemini_model
    try:
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured.")
            _gemini_configured = True
            return None
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        _gemini_configured = True
        logger.info(f"Gemini API configured (model: {GEMINI_MODEL}).")
        return _gemini_model
    except Exception as e:
        logger.error(f"Gemini configuration failed: {e}")
        _gemini_configured = True
        return None


def _detect_conflict(emotions: dict) -> str:
    """
    Detect conflicting emotional signals across modalities.
    Returns a conflict advisory string for the LLM, or empty string.
    """
    positive_labels = {"No Stress", "Happy", "Neutral", "Low Stress", "Calm"}
    negative_labels = {"STRESS DETECTED", "High Stress", "Angry", "Fear", "Sad", "Stressed"}

    has_positive = False
    has_negative = False
    positive_mods = []
    negative_mods = []

    for modality, result in emotions.items():
        label = result.get("label", "")
        if label in positive_labels:
            has_positive = True
            positive_mods.append(modality)
        elif label in negative_labels:
            has_negative = True
            negative_mods.append(modality)

    if has_positive and has_negative:
        return (
            f" ⚠️ CONFLICT DETECTED: The user's {', '.join(positive_mods)} "
            f"signal(s) appear positive, but their {', '.join(negative_mods)} "
            f"signal(s) show distress. Gently and compassionately check in "
            f"on this discrepancy — the user may be masking their feelings."
        )
    return ""


def _build_system_prompt(
    emotions: dict,
    communication_style: str = "Gentle & Reassuring",
) -> str:
    """Build the system instruction for Gemini from detected emotions."""
    system_parts = [
        f"You are Elysia, a compassionate AI mental-health companion.",
        f"Communicate in a **{communication_style}** tone.",
        "Keep responses concise (2-4 sentences), supportive, and safe. Formatted for a chat UI.",
        "Never diagnose, prescribe, or replace professional help.",
        "You remember the conversation context provided to you.",
    ]

    # Append detected emotion context
    context_lines = []
    for modality, result in emotions.items():
        label = result.get("label", "unknown")
        conf = result.get("confidence", 0)
        if modality == "text":
            context_lines.append(f"Text analysis indicates **{label}** (confidence {conf:.0%}).")
        elif modality == "voice":
            context_lines.append(f"Voice tonality (Wav2Vec2) indicates **{label}** (confidence {conf:.0%}).")
        elif modality == "face":
            context_lines.append(f"Facial expression (MobileNetV2) indicates **{label}** (confidence {conf:.0%}).")

    if context_lines:
        system_parts.append("Emotional context from user's multimodal inputs:")
        system_parts.extend(context_lines)
    else:
        system_parts.append("No strong emotion was detected from available inputs.")

    # Conflict detection
    conflict = _detect_conflict(emotions)
    if conflict:
        system_parts.append(conflict)

    return "\n".join(system_parts)


class _RateLimitError(Exception):
    """Custom exception for Gemini rate-limit (429) errors."""
    pass


class _NonRetryableError(Exception):
    """Custom exception for errors that should not be retried."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RateLimitError),
    reraise=True,
)
async def _try_gemini_call_with_retry(model_name: str, system_prompt: str, user_message: str, history: list) -> str:
    """Attempt a Gemini API call with exponential backoff retry on rate limits."""
    try:
        chat_model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
        )
        chat = chat_model.start_chat(history=history)
        response = chat.send_message(user_message)
        return response.text.strip()
    except Exception as e:
        error_str = str(e).lower()
        is_rate_limit = any(keyword in error_str for keyword in ["429", "quota", "rate", "resourceexhausted"])

        if is_rate_limit:
            logger.warning(f"Rate limited on {model_name}: {type(e).__name__}")
            raise _RateLimitError(str(e)) from e
        else:
            logger.error(f"Gemini error on {model_name}: {type(e).__name__}: {e}")
            raise _NonRetryableError(str(e)) from e


async def generate_response(
    emotions: dict,
    user_text: str | None,
    communication_style: str = "Gentle & Reassuring",
    chat_history: list | None = None,
) -> str:
    """
    Build a dynamic system prompt from detected emotions and invoke Gemini
    with conversation history for context-aware responses.
    Uses Tenacity for exponential backoff retry + model fallback.
    """
    system_prompt = _build_system_prompt(emotions, communication_style)
    user_message = user_text if user_text else "[User sent audio/image input only — no text.]"

    # Build chat history for context windowing (increased to 10 messages)
    history = []
    if chat_history:
        for msg in chat_history[-10:]:
            role = "user" if msg.get("role") == "user" else "model"
            history.append({"role": role, "parts": [msg.get("text", "")]})

    # ── Invoke Gemini with retry + fallback ──
    model = _ensure_gemini()
    if model is None:
        return LLM_FALLBACK + " (Gemini API key not configured.)"

    # Try primary model first, then fallback model
    models_to_try = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
    last_error = None

    for model_name in models_to_try:
        try:
            result = await _try_gemini_call_with_retry(model_name, system_prompt, user_message, history)
            if model_name != GEMINI_MODEL:
                logger.info(f"Gemini succeeded with fallback model: {model_name}")
            return result
        except _RateLimitError as e:
            last_error = e
            logger.warning(f"All retries exhausted for {model_name}, trying next model...")
            continue
        except _NonRetryableError as e:
            last_error = e
            logger.warning(f"Non-retryable error on {model_name}, trying next model...")
            continue
        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error on {model_name}: {e}")
            continue

    # All attempts failed
    logger.error(f"All Gemini models failed. Last error: {last_error}")
    return LLM_FALLBACK
