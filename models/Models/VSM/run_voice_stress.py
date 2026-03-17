"""
run_voice_stress.py — Voice Stress Detection Demo
===================================================
Provide a .wav audio file (or record from mic) and the
Wav2Vec2 model predicts stress level. Generates a
professional results panel saved as 'voice_stress_result.png'.

Usage:
    python run_voice_stress.py --audio sample.wav
    python run_voice_stress.py --record 5          # record 5 seconds from mic
"""

import os
import sys
import warnings
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress TF/HF noise
tf.get_logger().setLevel("ERROR")

from transformers import TFWav2Vec2Model, Wav2Vec2Config
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ================= CONFIGURATION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "finetuned_best.weights.h5")
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 5
TARGET_LEN = SAMPLE_RATE * MAX_DURATION_SEC   # 80000
NUM_CLASSES = 3
CLASS_LABELS = ["No Stress", "Low Stress", "High Stress"]
BAR_COLORS = ["#2ECC71", "#F39C12", "#E74C3C"]   # Green, Orange, Red
EMOJI_MAP = ["😊", "😐", "😰"]
BG_COLOR = "#1a1a2e"
CARD_COLOR = "#16213e"
ACCENT = "#0f3460"
TEXT_COLOR = "#e0e0e0"
TITLE_COLOR = "#00d4ff"
UNFREEZE_LAST_N = 4
# =================================================


def build_model():
    """Rebuild the exact Wav2Vec2 + attention-pooling architecture."""
    input_audio = tf.keras.Input(
        shape=(TARGET_LEN,), dtype=tf.float32, name="audio_input"
    )

    cfg = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    cfg.output_hidden_states = False

    backbone = TFWav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h", config=cfg, from_pt=True
    )

    # Match the training freeze pattern
    backbone.trainable = True
    try:
        encoder_layers = backbone.wav2vec2.encoder.layers
        num_layers = len(encoder_layers)
        freeze_until = num_layers - UNFREEZE_LAST_N
        for i in range(freeze_until):
            encoder_layers[i].trainable = False
        backbone.wav2vec2.feature_extractor.trainable = False
    except Exception:
        backbone.trainable = True

    # Dummy forward pass to build
    _ = backbone(tf.zeros((1, TARGET_LEN), dtype=tf.float32), training=False)

    outputs = backbone(input_audio, training=False)
    hidden = outputs.last_hidden_state   # (B, T, 768)

    # Attention pooling
    attn_scores = tf.keras.layers.Dense(1, name="attn_score")(hidden)
    attn_weights = tf.keras.layers.Softmax(axis=1, name="attn_weights")(attn_scores)
    x = tf.reduce_sum(hidden * attn_weights, axis=1)   # (B, 768)

    # Classification head
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", dtype="float32", name="predictions"
    )(x)

    return tf.keras.Model(inputs=input_audio, outputs=out)


def load_audio(path):
    """Load and preprocess audio to the expected format."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Pad or truncate
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)), "constant")
    else:
        audio = audio[:TARGET_LEN]

    # Normalize
    std = np.std(audio)
    if std > 0:
        audio = (audio - np.mean(audio)) / std

    return audio.astype(np.float32)


def record_audio(duration_sec):
    """Record audio from microphone using sounddevice."""
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"🎙️  Recording {duration_sec} seconds … speak now!")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()
    print("✅ Recording complete.")
    return audio


def generate_results_panel(audio_raw, probs, audio_path, output_path):
    """Create a professional, PPT-ready results panel."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    })

    pred_class = np.argmax(probs)
    confidences = probs * 100

    fig = plt.figure(figsize=(14, 7), facecolor=BG_COLOR)
    fig.suptitle(
        "🎤  Voice Stress Detection — Wav2Vec2",
        fontsize=20, fontweight="bold", color=TITLE_COLOR, y=0.97,
    )

    # --- Top panel: audio waveform ---
    ax_wave = fig.add_axes([0.06, 0.52, 0.88, 0.36])
    ax_wave.set_facecolor(CARD_COLOR)

    time_axis = np.linspace(0, len(audio_raw) / SAMPLE_RATE, len(audio_raw))
    ax_wave.fill_between(time_axis, audio_raw, alpha=0.35, color=BAR_COLORS[pred_class])
    ax_wave.plot(time_axis, audio_raw, color=BAR_COLORS[pred_class], linewidth=0.6)
    ax_wave.set_xlim(0, time_axis[-1])
    ax_wave.set_title(
        f"Audio Waveform  —  {os.path.basename(audio_path)}",
        fontsize=13, color=TEXT_COLOR, pad=8,
    )
    ax_wave.set_xlabel("Time (s)", fontsize=10, color=TEXT_COLOR)
    ax_wave.set_ylabel("Amplitude", fontsize=10, color=TEXT_COLOR)
    ax_wave.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax_wave.spines.values():
        spine.set_color(ACCENT)

    # --- Bottom-left: spectrogram ---
    ax_spec = fig.add_axes([0.06, 0.10, 0.42, 0.32])
    ax_spec.set_facecolor(CARD_COLOR)

    S = librosa.feature.melspectrogram(y=audio_raw, sr=SAMPLE_RATE, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = ax_spec.imshow(
        S_db, aspect="auto", origin="lower", cmap="magma",
        extent=[0, MAX_DURATION_SEC, 0, SAMPLE_RATE // 2],
    )
    ax_spec.set_title("Mel Spectrogram", fontsize=13, color=TEXT_COLOR, pad=8)
    ax_spec.set_xlabel("Time (s)", fontsize=10, color=TEXT_COLOR)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10, color=TEXT_COLOR)
    ax_spec.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax_spec.spines.values():
        spine.set_color(ACCENT)

    # --- Bottom-right: confidence bar chart ---
    ax_bar = fig.add_axes([0.56, 0.10, 0.38, 0.32])
    ax_bar.set_facecolor(CARD_COLOR)

    bars = ax_bar.barh(
        CLASS_LABELS, confidences, color=BAR_COLORS,
        edgecolor="white", linewidth=0.5, height=0.45,
    )

    bars[pred_class].set_edgecolor(TITLE_COLOR)
    bars[pred_class].set_linewidth(2.5)

    for j, (bar, conf) in enumerate(zip(bars, confidences)):
        ax_bar.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center", fontsize=11, fontweight="bold",
            color="white" if j == pred_class else TEXT_COLOR,
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_title("Stress Confidence", fontsize=13, color=TEXT_COLOR, pad=8)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_color(TEXT_COLOR)
    ax_bar.spines["left"].set_color(TEXT_COLOR)

    # --- Bottom banner ---
    result_text = (
        f"{EMOJI_MAP[pred_class]}  Prediction:  {CLASS_LABELS[pred_class]}   |   "
        f"Confidence: {confidences[pred_class]:.1f}%"
    )
    fig.text(
        0.50, 0.02, result_text,
        ha="center", fontsize=15, fontweight="bold",
        color=BAR_COLORS[pred_class],
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor=CARD_COLOR,
            edgecolor=BAR_COLORS[pred_class], linewidth=2,
        ),
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"✅ Results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Voice Stress Detection Demo")
    parser.add_argument("--audio", type=str, default=None, help="Path to a .wav audio file")
    parser.add_argument("--record", type=int, default=None, help="Record N seconds from mic")
    args = parser.parse_args()

    # 1. Build & load model
    print("🏗️  Building Wav2Vec2 model (downloading backbone if first run) …")
    model = build_model()
    print(f"🔄 Loading fine-tuned weights: {WEIGHTS_PATH}")
    try:
        model.load_weights(WEIGHTS_PATH)
        print("✅ Weights loaded!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Get audio
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"❌ File not found: {args.audio}")
            return
        audio_path = args.audio
        print(f"🎵 Loading audio: {audio_path}")
        audio = load_audio(audio_path)
    elif args.record:
        audio_raw = record_audio(args.record)
        # Save recording temporarily
        audio_path = os.path.join(SCRIPT_DIR, "_recorded.wav")
        import soundfile as sf
        sf.write(audio_path, audio_raw, SAMPLE_RATE)
        audio = load_audio(audio_path)
    else:
        # Look for any .wav in the directory
        wav_files = [f for f in os.listdir(SCRIPT_DIR) if f.endswith(".wav")]
        if wav_files:
            audio_path = os.path.join(SCRIPT_DIR, wav_files[0])
            print(f"🎵 Auto-detected: {audio_path}")
            audio = load_audio(audio_path)
        else:
            print("❌ No audio source specified!")
            print("   Use:  --audio file.wav   or   --record 5")
            return

    # 3. Predict
    input_tensor = np.expand_dims(audio, axis=0)
    print("🔍 Running inference …")
    predictions = model.predict(input_tensor, verbose=0)
    probs = predictions[0]
    pred_class = np.argmax(probs)

    print(f"\n{'='*45}")
    print(f"  {EMOJI_MAP[pred_class]}  Result: {CLASS_LABELS[pred_class]}")
    print(f"     Confidence: {probs[pred_class]*100:.1f}%")
    for i, label in enumerate(CLASS_LABELS):
        bar = "█" * int(probs[i] * 30) + "░" * (30 - int(probs[i] * 30))
        print(f"     {label:12s} [{bar}] {probs[i]*100:.1f}%")
    print(f"{'='*45}\n")

    # 4. Generate results panel
    # Use the raw audio (before normalization) for visualization
    raw_audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
    if len(raw_audio) < TARGET_LEN:
        raw_audio = np.pad(raw_audio, (0, TARGET_LEN - len(raw_audio)), "constant")
    else:
        raw_audio = raw_audio[:TARGET_LEN]

    output_path = os.path.join(SCRIPT_DIR, "voice_stress_result.png")
    generate_results_panel(raw_audio, probs, audio_path, output_path)


if __name__ == "__main__":
    main()
