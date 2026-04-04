"""
ml_service.py — Machine Learning model loading and inference.

Corrected architecture based on actual model analysis:
  • FSM: MobileNetV2 → 64×64 RGB → 3 stress classes
  • VSM: Wav2Vec2 TFLite → raw waveform (80 000 samples) → 3 classes
  • Text: BertForSequenceClassification → tokenized text → 2 classes

CRITICAL FIX: The voice model expects RAW WAVEFORMS, not MFCCs.

Quantized model support:
  • Text: Prefers ONNX INT8 model if available, falls back to TF BERT
  • Face: Prefers TFLite model if available, falls back to full Keras

All heavy ML imports are LAZY — server starts even if TF isn't installed.
"""
import io
import os

from backend.config import (
    FSM_MODEL_PATH, FSM_TFLITE_PATH, FSM_IMG_SIZE, FSM_CLASSES,
    VSM_MODEL_PATH, VSM_SAMPLE_RATE, VSM_TARGET_LEN, VSM_CLASSES,
    TEXT_MODEL_DIR, TEXT_ONNX_PATH, TEXT_CLASSES, TEXT_MAX_LEN,
)
from backend.logger import get_logger

logger = get_logger(__name__)


class MLService:
    """Holds all three ML models after async initialization."""

    def __init__(self):
        self.fer_model = None         # Face (Keras — legacy)
        self.fer_interpreter = None   # Face (TFLite — preferred)
        self.voice_interpreter = None  # Voice (TFLite)
        self.text_tokenizer = None    # Text tokenizer
        self.text_model = None        # Text model (TF — legacy)
        self.text_onnx_session = None # Text model (ONNX — preferred)
        self.face_cascade = None      # Haar Cascade for face detection
        self._use_onnx_text = False   # Flag: using ONNX for text?
        self._use_tflite_face = False # Flag: using TFLite for face?

    # ═══════════════════════════════════════════════
    #  MODEL LOADING (called in lifespan)
    # ═══════════════════════════════════════════════

    def load_face_model(self):
        """Load FSM: Prefer TFLite, fall back to Keras MobileNetV2."""
        # Try TFLite first (quantized, faster)
        if os.path.exists(FSM_TFLITE_PATH):
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=FSM_TFLITE_PATH)
                interpreter.allocate_tensors()
                self.fer_interpreter = interpreter
                self._use_tflite_face = True

                # Still need Haar Cascade for face detection
                import cv2
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

                logger.info("Face Stress Model (TFLite — quantized) loaded.")
                return
            except Exception as e:
                logger.warning(f"TFLite face model failed, trying Keras: {e}")

        # Fall back to full Keras model
        try:
            import numpy as np
            import cv2
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import (
                GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
            )
            from tensorflow.keras.models import Sequential

            if not os.path.exists(FSM_MODEL_PATH):
                logger.warning(f"FSM model file not found: {FSM_MODEL_PATH}")
                return

            # Rebuild architecture to match training
            base = MobileNetV2(weights=None, include_top=False, input_shape=(64, 64, 3))
            base.trainable = True
            model = Sequential([
                base,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(3, activation="softmax"),
            ])
            model.load_weights(FSM_MODEL_PATH)
            self.fer_model = model

            # Load Haar Cascade for pre-screening
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            logger.info("Face Stress Model (MobileNetV2 — Keras) loaded.")
        except ImportError as ie:
            logger.warning(f"FSM skipped — missing dependency: {ie}")
        except Exception as e:
            logger.error(f"FSM load failed: {e}")

    def load_voice_model(self):
        """Load VSM: Wav2Vec2 fine-tuned TFLite interpreter."""
        try:
            import tensorflow as tf

            if not os.path.exists(VSM_MODEL_PATH):
                logger.warning(f"VSM model file not found: {VSM_MODEL_PATH}")
                return

            interpreter = tf.lite.Interpreter(model_path=VSM_MODEL_PATH)
            interpreter.allocate_tensors()
            self.voice_interpreter = interpreter
            logger.info("Voice Stress Model (TFLite) loaded.")
        except ImportError as ie:
            logger.warning(f"VSM skipped — missing dependency: {ie}")
        except Exception as e:
            logger.error(f"VSM load failed: {e}")

    def load_text_model(self):
        """Load Text: Prefer ONNX INT8, fall back to TF BERT."""
        # Try ONNX first (quantized, faster, smaller)
        if os.path.exists(TEXT_ONNX_PATH):
            try:
                import onnxruntime as ort
                from transformers import BertTokenizer

                self.text_onnx_session = ort.InferenceSession(TEXT_ONNX_PATH)
                self.text_tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_DIR)
                self._use_onnx_text = True
                logger.info("Text Stress Model (ONNX INT8 — quantized) loaded.")
                return
            except ImportError:
                logger.warning("onnxruntime not installed, falling back to TF BERT.")
            except Exception as e:
                logger.warning(f"ONNX text model failed, trying TF BERT: {e}")

        # Fall back to TF BERT
        try:
            if not os.path.exists(os.path.join(TEXT_MODEL_DIR, "tf_model.h5")):
                logger.warning(f"Text model file not found in: {TEXT_MODEL_DIR}")
                return

            from transformers import BertTokenizer, TFBertForSequenceClassification
            self.text_tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_DIR)
            self.text_model = TFBertForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
            logger.info("Text Stress Model (BERT — TF) loaded.")
        except ImportError as ie:
            logger.warning(f"Text model skipped — missing dependency: {ie}")
        except Exception as e:
            logger.error(f"Text model load failed: {e}")

    # ═══════════════════════════════════════════════
    #  INFERENCE METHODS
    # ═══════════════════════════════════════════════

    def predict_text(self, text: str) -> dict:
        """
        Run text stress classification.
        Prefers ONNX if available, falls back to TF BERT.
        Returns: {"label": str, "confidence": float}
        """
        # ONNX path
        if self._use_onnx_text and self.text_onnx_session and self.text_tokenizer:
            return self._predict_text_onnx(text)

        # TF BERT path
        if self.text_model and self.text_tokenizer:
            return self._predict_text_tf(text)

        return {"label": "unavailable", "confidence": 0.0}

    def _predict_text_onnx(self, text: str) -> dict:
        """ONNX Runtime inference for text classification."""
        import numpy as np

        tokens = self.text_tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=TEXT_MAX_LEN,
        )

        # Build feed dict matching ONNX input names
        session = self.text_onnx_session
        input_names = [inp.name for inp in session.get_inputs()]
        feed = {}
        for name in input_names:
            if "input_ids" in name:
                feed[name] = tokens["input_ids"].astype(np.int32)
            elif "attention_mask" in name:
                feed[name] = tokens["attention_mask"].astype(np.int32)
            elif "token_type_ids" in name:
                feed[name] = tokens["token_type_ids"].astype(np.int32)

        outputs = session.run(None, feed)
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        idx = int(np.argmax(probs))
        return {"label": TEXT_CLASSES[idx], "confidence": float(probs[idx])}

    def _predict_text_tf(self, text: str) -> dict:
        """TensorFlow BERT inference for text classification."""
        import numpy as np
        import tensorflow as tf

        inputs = self.text_tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=TEXT_MAX_LEN,
        )
        outputs = self.text_model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": TEXT_CLASSES[idx], "confidence": float(probs[idx])}

    def predict_face(self, image_bytes: bytes) -> dict:
        """
        Run face stress classification on an image.
        Steps: decode → detect face (Haar) → resize 64×64 → normalize → predict.
        Returns: {"label": str, "confidence": float} or error dict.
        """
        if not self.fer_model and not self.fer_interpreter:
            return {"label": "unavailable", "confidence": 0.0}

        import numpy as np
        import cv2

        # Decode image from bytes
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"label": "error", "confidence": 0.0, "detail": "Could not decode image."}

        # Face detection (edge case: no face → soft error)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return {"label": "no_face_detected", "confidence": 0.0,
                    "detail": "No face detected. Please try again with better lighting."}

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = img[y:y+h, x:x+w]

        # Preprocess exactly as Run_realtime.py does
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_face, FSM_IMG_SIZE)
        normalized = resized / 255.0
        tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

        # TFLite path (preferred)
        if self._use_tflite_face and self.fer_interpreter:
            return self._predict_face_tflite(tensor)

        # Keras path (legacy)
        preds = self.fer_model.predict(tensor, verbose=0)
        idx = int(np.argmax(preds))
        return {"label": FSM_CLASSES[idx], "confidence": float(np.max(preds))}

    def _predict_face_tflite(self, tensor) -> dict:
        """TFLite inference for face stress classification."""
        import numpy as np

        interpreter = self.fer_interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Resize input tensor if needed
        if list(input_details[0]["shape"]) != list(tensor.shape):
            interpreter.resize_tensor_input(input_details[0]["index"], tensor.shape)
            interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]["index"], tensor)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]

        idx = int(np.argmax(preds))
        return {"label": FSM_CLASSES[idx], "confidence": float(preds[idx])}

    def predict_voice(self, audio_bytes: bytes) -> dict:
        """
        Run voice stress classification.
        CRITICAL: This model expects RAW WAVEFORMS (80 000 float32 samples at 16 kHz),
                  NOT MFCCs. The previous implementation was incorrect.
        Steps: load audio → resample 16 kHz → pad/truncate to 80 000 → normalize → TFLite.
        """
        if not self.voice_interpreter:
            return {"label": "unavailable", "confidence": 0.0}

        try:
            import numpy as np
            import librosa

            # Load audio from bytes
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=VSM_SAMPLE_RATE,
                                    duration=5)

            # Edge case: silent audio
            if np.max(np.abs(audio)) < 1e-4:
                return {"label": "silent_audio", "confidence": 0.0,
                        "detail": "Audio is nearly silent. Please try with louder input."}

            # Mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Pad or truncate to 80 000 samples (exactly as run_voice_stress.py does)
            if len(audio) < VSM_TARGET_LEN:
                audio = np.pad(audio, (0, VSM_TARGET_LEN - len(audio)), "constant")
            else:
                audio = audio[:VSM_TARGET_LEN]

            # Normalize (zero mean, unit std)
            std = np.std(audio)
            if std > 0:
                audio = (audio - np.mean(audio)) / std

            # Shape: (1, 80000) — batch of 1
            input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)

            # TFLite inference
            interpreter = self.voice_interpreter
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Resize input tensor if the TFLite model has dynamic shape
            if list(input_details[0]["shape"]) != list(input_tensor.shape):
                interpreter.resize_tensor_input(input_details[0]["index"], input_tensor.shape)
                interpreter.allocate_tensors()

            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            idx = int(np.argmax(preds))
            return {"label": VSM_CLASSES[idx], "confidence": float(preds[idx])}

        except ImportError as ie:
            return {"label": "error", "confidence": 0.0, "detail": f"Missing dependency: {ie}"}
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "detail": str(e)}


# Singleton
ml_service = MLService()
