"""
run_face_stress.py — Facial Stress Detection Demo
===================================================
Captures a photo from your webcam (or loads an image),
runs the MobileNetV2 stress model, and generates a
professional results panel saved as 'fsm_result.png'.

Usage:
    python run_face_stress.py                  # uses webcam
    python run_face_stress.py --image photo.jpg # uses an image file
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import sys

# ================= CONFIGURATION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_stress_model.keras")
IMG_SIZE = (64, 64)
CLASS_LABELS = ["High Stress", "Low Stress", "No Stress"]
BAR_COLORS = ["#E74C3C", "#F39C12", "#2ECC71"]   # Red, Orange, Green
BG_COLOR = "#1a1a2e"
CARD_COLOR = "#16213e"
ACCENT = "#0f3460"
TEXT_COLOR = "#e0e0e0"
TITLE_COLOR = "#00d4ff"
# =================================================


def build_model(input_shape=(64, 64, 3), num_classes=3):
    """Rebuild the exact same MobileNetV2 architecture for weight loading."""
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def capture_from_webcam():
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        sys.exit(1)

    print("📸 Webcam opened — press SPACE to capture, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    print("❌ Failed to capture frame.")
    sys.exit(1)


def detect_face(frame):
    """Detect the largest face in the frame."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        return None, None

    # Pick the largest face
    areas = [w * h for (_, _, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]
    face_roi = frame[y : y + h, x : x + w]
    return face_roi, (x, y, w, h)


def generate_results_panel(original_frame, face_roi, bbox, predictions, output_path):
    """Create a professional, PPT-ready results panel and save it."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    })

    fig = plt.figure(figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle(
        "🧠  Facial Stress Detection — MobileNetV2",
        fontsize=20, fontweight="bold", color=TITLE_COLOR, y=0.97,
    )

    # --- Left panel: detected face with bounding box ---
    ax1 = fig.add_axes([0.03, 0.08, 0.35, 0.78])
    ax1.set_facecolor(CARD_COLOR)

    display_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 212, 255), 3)

    ax1.imshow(display_frame)
    ax1.set_title("Detected Face", fontsize=13, color=TEXT_COLOR, pad=8)
    ax1.axis("off")

    # --- Middle panel: zoomed-in face ---
    ax2 = fig.add_axes([0.40, 0.20, 0.18, 0.60])
    ax2.set_facecolor(CARD_COLOR)
    if face_roi is not None:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        ax2.imshow(face_rgb)
    ax2.set_title("Face ROI", fontsize=13, color=TEXT_COLOR, pad=8)
    ax2.axis("off")

    # --- Right panel: prediction bar chart ---
    ax3 = fig.add_axes([0.64, 0.15, 0.33, 0.65])
    ax3.set_facecolor(CARD_COLOR)

    confidences = predictions[0] * 100
    pred_class = np.argmax(predictions[0])

    bars = ax3.barh(
        CLASS_LABELS, confidences, color=BAR_COLORS,
        edgecolor="white", linewidth=0.5, height=0.5,
    )

    # Highlight the winner
    bars[pred_class].set_edgecolor(TITLE_COLOR)
    bars[pred_class].set_linewidth(2.5)

    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax3.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center", fontsize=12, fontweight="bold",
            color="white" if i == pred_class else TEXT_COLOR,
        )

    ax3.set_xlim(0, 110)
    ax3.set_title("Stress Confidence", fontsize=13, color=TEXT_COLOR, pad=8)
    ax3.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color(TEXT_COLOR)
    ax3.spines["left"].set_color(TEXT_COLOR)
    ax3.xaxis.label.set_color(TEXT_COLOR)

    # --- Bottom banner: prediction result ---
    result_text = f"Prediction:  {CLASS_LABELS[pred_class]}   |   Confidence: {confidences[pred_class]:.1f}%"
    fig.text(
        0.50, 0.02, result_text,
        ha="center", fontsize=15, fontweight="bold",
        color=BAR_COLORS[pred_class],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_COLOR, edgecolor=BAR_COLORS[pred_class], linewidth=2),
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"✅ Results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Facial Stress Detection Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file (skip webcam)")
    args = parser.parse_args()

    # 1. Build & load model
    print("🏗️  Building model architecture …")
    model = build_model()
    print(f"🔄 Loading weights: {MODEL_PATH}")
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Get image
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Could not read image: {args.image}")
            return
        print(f"📷 Loaded image: {args.image}")
    else:
        frame = capture_from_webcam()

    # 3. Detect face
    face_roi, bbox = detect_face(frame)
    if face_roi is None:
        print("⚠️  No face detected — running prediction on full frame.")
        face_roi = frame
        bbox = None

    # 4. Preprocess & predict
    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_face, IMG_SIZE)
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    predictions = model.predict(input_tensor, verbose=0)
    pred_class = np.argmax(predictions[0])

    print(f"\n{'='*40}")
    print(f"  Result: {CLASS_LABELS[pred_class]}")
    print(f"  Confidence: {predictions[0][pred_class]*100:.1f}%")
    print(f"{'='*40}\n")

    # 5. Generate results panel
    output_path = os.path.join(SCRIPT_DIR, "fsm_result.png")
    generate_results_panel(frame, face_roi, bbox, predictions, output_path)


if __name__ == "__main__":
    main()
