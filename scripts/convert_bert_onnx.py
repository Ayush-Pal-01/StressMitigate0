"""
convert_bert_onnx.py — Convert TF BERT stress model to ONNX + INT8 quantization.

Usage:
    python scripts/convert_bert_onnx.py

Requirements:
    pip install tf2onnx onnxruntime onnx onnxruntime-extensions

Input:  models/Models/saved_stress_model/tf_model.h5
Output: models/Models/saved_stress_model/model_quantized.onnx

Expected size reduction: ~438MB → ~100MB
Expected speedup: 2-3x on CPU inference
"""
import os
import sys
import time
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Models", "saved_stress_model")
TF_MODEL_PATH = os.path.join(MODEL_DIR, "tf_model.h5")
ONNX_OUTPUT_PATH = os.path.join(MODEL_DIR, "model.onnx")
ONNX_QUANTIZED_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")


def step1_export_to_onnx():
    """Export TF BERT model to ONNX format."""
    print("\n═══ Step 1: Export TF BERT → ONNX ═══")

    if not os.path.exists(TF_MODEL_PATH):
        print(f"❌ Model file not found: {TF_MODEL_PATH}")
        print("   Make sure the BERT model weights are in the correct location.")
        sys.exit(1)

    print(f"   Loading model from: {TF_MODEL_PATH}")
    from transformers import TFBertForSequenceClassification
    model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)

    print(f"   Converting to ONNX...")
    import tf2onnx
    import tensorflow as tf

    # Define input signature for BERT
    input_spec = (
        tf.TensorSpec((1, 128), tf.int32, name="input_ids"),
        tf.TensorSpec((1, 128), tf.int32, name="attention_mask"),
        tf.TensorSpec((1, 128), tf.int32, name="token_type_ids"),
    )

    # Convert
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=13,
        output_path=ONNX_OUTPUT_PATH,
    )

    onnx_size = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
    print(f"   ✅ ONNX model saved: {ONNX_OUTPUT_PATH} ({onnx_size:.1f} MB)")
    return ONNX_OUTPUT_PATH


def step2_quantize_int8():
    """Apply INT8 dynamic quantization to the ONNX model."""
    print("\n═══ Step 2: INT8 Dynamic Quantization ═══")

    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(ONNX_OUTPUT_PATH):
        print(f"❌ ONNX model not found: {ONNX_OUTPUT_PATH}")
        print("   Run step 1 first.")
        sys.exit(1)

    print(f"   Quantizing {ONNX_OUTPUT_PATH}...")
    quantize_dynamic(
        model_input=ONNX_OUTPUT_PATH,
        model_output=ONNX_QUANTIZED_PATH,
        weight_type=QuantType.QInt8,
    )

    original_size = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(ONNX_QUANTIZED_PATH) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"   Original ONNX:   {original_size:.1f} MB")
    print(f"   Quantized INT8:  {quantized_size:.1f} MB")
    print(f"   Size reduction:  {reduction:.1f}%")
    print(f"   ✅ Quantized model saved: {ONNX_QUANTIZED_PATH}")


def step3_validate():
    """Validate quantized model produces same outputs as original."""
    print("\n═══ Step 3: Validation ═══")

    from transformers import BertTokenizer
    import onnxruntime as ort

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    # Test sentences
    test_texts = [
        "I'm feeling really stressed about my exams",
        "Today was a wonderful day, I feel great",
        "Work pressure is overwhelming me",
        "I had a relaxing walk in the park",
        "I can't sleep, everything feels wrong",
    ]

    # Load quantized ONNX model
    session = ort.InferenceSession(ONNX_QUANTIZED_PATH)
    input_names = [inp.name for inp in session.get_inputs()]

    print(f"   ONNX inputs: {input_names}")
    print(f"   Testing with {len(test_texts)} sentences...\n")

    classes = ["No Stress", "STRESS DETECTED"]

    total_time = 0
    for text in test_texts:
        tokens = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        # Build feed dict matching ONNX input names
        feed = {}
        for name in input_names:
            key = name.replace(".", "_")  # handle naming differences
            if "input_ids" in name:
                feed[name] = tokens["input_ids"].astype(np.int32)
            elif "attention_mask" in name:
                feed[name] = tokens["attention_mask"].astype(np.int32)
            elif "token_type_ids" in name:
                feed[name] = tokens["token_type_ids"].astype(np.int32)

        start = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
        idx = int(np.argmax(probs))

        print(f"   \"{text[:50]}...\"")
        print(f"   → {classes[idx]} (confidence: {probs[idx]:.2%}) [{elapsed:.0f}ms]")
        print()

    avg_time = total_time / len(test_texts)
    print(f"   Average inference time: {avg_time:.0f}ms per prediction")
    print(f"   ✅ Validation complete!")


def main():
    print("=" * 60)
    print("  BERT Stress Model → ONNX INT8 Conversion")
    print("=" * 60)

    tf_size = os.path.getsize(TF_MODEL_PATH) / (1024 * 1024) if os.path.exists(TF_MODEL_PATH) else 0
    print(f"\n   Source: {TF_MODEL_PATH}")
    print(f"   Size:   {tf_size:.1f} MB")

    step1_export_to_onnx()
    step2_quantize_int8()
    step3_validate()

    print("\n" + "=" * 60)
    print("  ✅ Conversion complete!")
    print(f"  Quantized model: {ONNX_QUANTIZED_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
