"""
convert_bert_onnx.py — Convert TF BERT stress model to ONNX + INT8 quantization.

Strategy: Use tf2onnx.convert.from_function() on a concrete function
from the TF BERT model. This avoids SavedModel file locking issues on
Windows and handles GELU activation properly with opset 15+.

Usage:
    python scripts/convert_bert_onnx.py

Requirements:
    pip install tf2onnx onnxruntime onnx transformers tensorflow
"""
import os
import sys
import time
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Models", "saved_stress_model")
ONNX_OUTPUT_PATH = os.path.join(MODEL_DIR, "model.onnx")
ONNX_QUANTIZED_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")
MAX_LEN = 128


def step1_export_to_onnx():
    """Export TF BERT model to ONNX via tf2onnx from_function."""
    print("\n═══ Step 1: Export TF BERT → ONNX ═══")

    import tensorflow as tf
    from transformers import TFBertForSequenceClassification

    print(f"   Loading TF BERT model from: {MODEL_DIR}")
    model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
    print(f"   ✅ Model loaded.")

    # Create concrete function with explicit input signature
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='attention_mask'),
        tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='token_type_ids'),
    ])
    def serve(input_ids, attention_mask, token_type_ids):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=False,
        )
        return output.logits

    # Convert using tf2onnx — pass the tf.function directly
    print(f"   Converting to ONNX (opset 15 — supports Erf/GELU)...")
    import tf2onnx

    model_proto, _ = tf2onnx.convert.from_function(
        serve,
        opset=15,
        output_path=ONNX_OUTPUT_PATH,
        input_signature=[
            tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='attention_mask'),
            tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32, name='token_type_ids'),
        ],
    )

    if os.path.exists(ONNX_OUTPUT_PATH):
        onnx_size = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
        print(f"   ✅ ONNX model saved: {ONNX_OUTPUT_PATH} ({onnx_size:.1f} MB)")
    else:
        print(f"   ❌ ONNX export failed.")
        sys.exit(1)


def step2_quantize_int8():
    """Apply INT8 dynamic quantization to the ONNX model."""
    print("\n═══ Step 2: INT8 Dynamic Quantization ═══")

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"   Quantizing...")
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

    # Clean up intermediate unquantized ONNX
    if os.path.exists(ONNX_OUTPUT_PATH):
        os.remove(ONNX_OUTPUT_PATH)
        print(f"   Cleaned up intermediate model.onnx")


def step3_validate():
    """Validate quantized model works correctly."""
    print("\n═══ Step 3: Validation ═══")

    from transformers import BertTokenizer
    import onnxruntime as ort

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    test_texts = [
        "I'm feeling really stressed about my exams",
        "Today was a wonderful day, I feel great",
        "Work pressure is overwhelming me",
        "I had a relaxing walk in the park",
        "I can't sleep, everything feels wrong",
    ]

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
            max_length=MAX_LEN,
        )

        feed = {}
        for name in input_names:
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
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        idx = int(np.argmax(probs))

        print(f"   \"{text[:50]}\"")
        print(f"   → {classes[idx]} (confidence: {probs[idx]:.2%}) [{elapsed:.0f}ms]")
        print()

    avg_time = total_time / len(test_texts)
    print(f"   Average inference time: {avg_time:.0f}ms per prediction")
    print(f"   ✅ Validation complete!")


def main():
    print("=" * 60)
    print("  BERT Stress Model → ONNX INT8 Conversion")
    print("=" * 60)

    tf_model_path = os.path.join(MODEL_DIR, "tf_model.h5")
    if os.path.exists(tf_model_path):
        tf_size = os.path.getsize(tf_model_path) / (1024 * 1024)
        print(f"\n   Source: {MODEL_DIR}")
        print(f"   tf_model.h5 size: {tf_size:.1f} MB")

    step1_export_to_onnx()
    step2_quantize_int8()
    step3_validate()

    final_size = os.path.getsize(ONNX_QUANTIZED_PATH) / (1024 * 1024)
    print("\n" + "=" * 60)
    print("  ✅ Conversion complete!")
    print(f"  Quantized model: {ONNX_QUANTIZED_PATH}")
    print(f"  Final size: {final_size:.1f} MB (was 417.9 MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
