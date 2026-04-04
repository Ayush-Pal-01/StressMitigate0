"""
test_brain.py — Tests for the brain pipeline (emotion fusion + Gemini).
"""
import pytest
from backend.brain import _detect_conflict, _build_system_prompt


class TestConflictDetection:
    """Test emotional conflict detection across modalities."""

    def test_no_conflict_all_positive(self):
        emotions = {
            "text": {"label": "No Stress", "confidence": 0.9},
            "voice": {"label": "Calm", "confidence": 0.85},
        }
        result = _detect_conflict(emotions)
        assert result == ""

    def test_no_conflict_all_negative(self):
        emotions = {
            "text": {"label": "STRESS DETECTED", "confidence": 0.95},
            "voice": {"label": "High Stress", "confidence": 0.88},
        }
        result = _detect_conflict(emotions)
        assert result == ""

    def test_conflict_detected(self):
        """If text says calm but face shows stress, flag a conflict."""
        emotions = {
            "text": {"label": "No Stress", "confidence": 0.9},
            "face": {"label": "Stressed", "confidence": 0.85},
        }
        result = _detect_conflict(emotions)
        assert "CONFLICT DETECTED" in result
        assert "text" in result
        assert "face" in result

    def test_conflict_with_masking(self):
        """User's text is positive, voice shows distress."""
        emotions = {
            "text": {"label": "No Stress", "confidence": 0.92},
            "voice": {"label": "High Stress", "confidence": 0.87},
        }
        result = _detect_conflict(emotions)
        assert "CONFLICT DETECTED" in result
        assert "masking" in result.lower()

    def test_empty_emotions(self):
        result = _detect_conflict({})
        assert result == ""


class TestSystemPromptBuilder:
    """Test the system prompt construction."""

    def test_basic_prompt_structure(self):
        emotions = {"text": {"label": "Low Stress", "confidence": 0.75}}
        prompt = _build_system_prompt(emotions)
        assert "Elysia" in prompt
        assert "Gentle & Reassuring" in prompt
        assert "Low Stress" in prompt
        assert "75%" in prompt

    def test_custom_communication_style(self):
        emotions = {}
        prompt = _build_system_prompt(emotions, communication_style="Direct & Professional")
        assert "Direct & Professional" in prompt

    def test_no_emotions(self):
        prompt = _build_system_prompt({})
        assert "No strong emotion" in prompt

    def test_all_three_modalities(self):
        emotions = {
            "text": {"label": "High Stress", "confidence": 0.98},
            "voice": {"label": "Low Stress", "confidence": 0.72},
            "face": {"label": "Stressed", "confidence": 0.85},
        }
        prompt = _build_system_prompt(emotions)
        assert "Text analysis" in prompt
        assert "Voice tonality" in prompt
        assert "Facial expression" in prompt
