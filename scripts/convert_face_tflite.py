"""
convert_face_tflite.py — Convert MobileNetV2 face stress model (.keras) to TFLite.

Usage:
    python scripts/convert_face_tflite.py

Input:  models/Models/FSM/fine_tuned_stress_model.keras
Output: models/Models/FSM/face_stress.tflite

Expected size reduction: ~26MB → ~10MB
Unifies inference pipeline (Voice model already uses TFLite).
"""
import os
import sys
import time
import numpy as np


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FSM_DIR = os.path.join(PROJECT_ROOT, "models", "Models", "FSM")
KERAS_MODEL_PATH = os.path.join(FSM_DIR, "fine_tuned_stress_model.keras")
TFLITE_OUTPUT_PATH = os.path.join(FSM_DIR, "face_stress.tflite")

FSM_CLASSES = ["High Stress", "Low Stress", "No Stress"]
IMG_SIZE = (64, 64)


def step1_convert_to_tflite():
    """Convert .keras model to TFLite with float16 quantization."""
    print("\n═══ Step 1: Convert MobileNetV2 (.keras) → TFLite ═══")

    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ Model file not found: {KERAS_MODEL_PATH}")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.models import Sequential

    print(f"   Rebuilding architecture...")

    # Rebuild the exact architecture from ml_service.py
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

    print(f"   Loading weights from: {KERAS_MODEL_PATH}")
    model.load_weights(KERAS_MODEL_PATH)

    # Build the model by running a dummy input
    dummy_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
    _ = model.predict(dummy_input, verbose=0)

    print(f"   Converting to TFLite with float16 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size = len(tflite_model) / (1024 * 1024)
    reduction = (1 - tflite_size / keras_size) * 100

    print(f"   Original .keras:  {keras_size:.1f} MB")
    print(f"   TFLite float16:   {tflite_size:.1f} MB")
    print(f"   Size reduction:   {reduction:.1f}%")
    print(f"   ✅ TFLite model saved: {TFLITE_OUTPUT_PATH}")


def step2_validate():
    """Validate TFLite model produces same outputs as original."""
    print("\n═══ Step 2: Validation ═══")

    import tensorflow as tf

    if not os.path.exists(TFLITE_OUTPUT_PATH):
        print(f"❌ TFLite model not found: {TFLITE_OUTPUT_PATH}")
        sys.exit(1)

    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")

    # Test with random face-like images (normalized 0-1, 64x64 RGB)
    print(f"\n   Running 5 test inferences...")
    total_time = 0

    for i in range(5):
        test_img = np.random.rand(1, 64, 64, 3).astype(np.float32)

        # Resize input if needed
        if list(input_details[0]["shape"]) != list(test_img.shape):
            interpreter.resize_tensor_input(input_details[0]["index"], test_img.shape)
            interpreter.allocate_tensors()

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], test_img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        idx = int(np.argmax(preds))
        print(f"   Test {i+1}: {FSM_CLASSES[idx]} (conf: {preds[idx]:.2%}) [{elapsed:.0f}ms]")

    avg_time = total_time / 5
    print(f"\n   Average inference time: {avg_time:.0f}ms")
    print(f"   ✅ Validation complete!")


def main():
    print("=" * 60)
    print("  Face Stress Model → TFLite Conversion")
    print("=" * 60)

    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024) if os.path.exists(KERAS_MODEL_PATH) else 0
    print(f"\n   Source: {KERAS_MODEL_PATH}")
    print(f"   Size:   {keras_size:.1f} MB")

    step1_convert_to_tflite()
    step2_validate()

    print("\n" + "=" * 60)
    print("  ✅ Conversion complete!")
    print(f"  TFLite model: {TFLITE_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
