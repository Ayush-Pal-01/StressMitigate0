"""
test_ml_service.py — Tests for the ML service layer.
"""
import pytest
from backend.ml_service import MLService


class TestMLServiceFallbacks:
    """Test that ML service returns 'unavailable' when models aren't loaded."""

    def test_predict_text_no_model(self):
        """Text prediction should return unavailable when no model loaded."""
        service = MLService()
        result = service.predict_text("I am stressed")
        assert result["label"] == "unavailable"
        assert result["confidence"] == 0.0

    def test_predict_face_no_model(self):
        """Face prediction should return unavailable when no model loaded."""
        service = MLService()
        result = service.predict_face(b"fake image bytes")
        assert result["label"] == "unavailable"
        assert result["confidence"] == 0.0

    def test_predict_voice_no_model(self):
        """Voice prediction should return unavailable when no model loaded."""
        service = MLService()
        result = service.predict_voice(b"fake audio bytes")
        assert result["label"] == "unavailable"
        assert result["confidence"] == 0.0


class TestMLServiceInit:
    """Test MLService initialization state."""

    def test_initial_state(self):
        """All models should be None on init."""
        service = MLService()
        assert service.fer_model is None
        assert service.fer_interpreter is None
        assert service.voice_interpreter is None
        assert service.text_model is None
        assert service.text_onnx_session is None
        assert service.text_tokenizer is None
        assert service._use_onnx_text is False
        assert service._use_tflite_face is False


class TestTextStressClassification:
    """Test the 3-tier text stress classification logic."""

    def test_classify_high_stress(self):
        """STRESS DETECTED with >97.5% confidence → High Stress."""
        from backend.routers.chat_router import _classify_text_stress
        result = _classify_text_stress({"label": "STRESS DETECTED", "confidence": 0.99})
        assert result["label"] == "High Stress"

    def test_classify_low_stress_from_detected(self):
        """STRESS DETECTED with <97.5% confidence → Low Stress."""
        from backend.routers.chat_router import _classify_text_stress
        result = _classify_text_stress({"label": "STRESS DETECTED", "confidence": 0.90})
        assert result["label"] == "Low Stress"

    def test_classify_no_stress_high_confidence(self):
        """No Stress with >84% confidence → No Stress."""
        from backend.routers.chat_router import _classify_text_stress
        result = _classify_text_stress({"label": "No Stress", "confidence": 0.95})
        assert result["label"] == "No Stress"

    def test_classify_no_stress_low_confidence(self):
        """No Stress with <=84% confidence → Low Stress (uncertain)."""
        from backend.routers.chat_router import _classify_text_stress
        result = _classify_text_stress({"label": "No Stress", "confidence": 0.75})
        assert result["label"] == "Low Stress"
