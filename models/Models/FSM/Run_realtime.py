import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import time
import os

# ================= CONFIGURATION =================
MODEL_PATH = 'fine_tuned_stress_model.keras'
IMG_SIZE = (64, 64)
CLASS_LABELS = ['High Stress', 'Low Stress', 'No Stress']
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0)]
# =================================================

def build_model(input_shape=(64, 64, 3), num_classes=3):
    """
    Re-builds the exact same architecture so we can load weights safely.
    """
    base_model = MobileNetV2(
        weights=None,            # We don't need ImageNet weights, we'll load our own
        include_top=False, 
        input_shape=input_shape
    )
    base_model.trainable = True  # Must match the fine-tuned state

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def load_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Error: Could not load face detector XML.")
        exit()
    return face_cascade

def preprocess_face(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized_face = cv2.resize(rgb_face, IMG_SIZE)
    normalized_face = resized_face / 255.0
    reshaped_face = np.expand_dims(normalized_face, axis=0)
    return reshaped_face

def main():
    # 1. Build the empty structure first
    print("🏗️ Re-building model architecture...")
    model = build_model()
    
    # 2. Load the weights specifically
    print(f"🔄 Loading weights from: {MODEL_PATH}...")
    try:
        # We use .load_weights instead of .load_model
        model.load_weights(MODEL_PATH)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Try Option 2 below if this fails.")
        return

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    face_cascade = load_face_detector()
    
    print("🚀 Starting Real-Time Stress Detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                processed_face = preprocess_face(face_roi)
                
                # Predict
                predictions = model.predict(processed_face, verbose=0)
                class_index = np.argmax(predictions)
                confidence = np.max(predictions)
                
                label_text = f"{CLASS_LABELS[class_index]} ({confidence*100:.1f}%)"
                color = COLORS[class_index]

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception:
                pass

        cv2.imshow('Stress Detection System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    




''' import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# ================= CONFIGURATION =================
MODEL_PATH = 'fine_tuned_stress_model.keras'
IMG_SIZE = (64, 64)
CLASS_LABELS = ['High Stress', 'Low Stress', 'No Stress']
# Colors for the boxes: Red, Orange, Green (in BGR format for OpenCV)
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0)]
# =================================================

def load_face_detector():
    # Load the standard Haar Cascade for face detection
    # OpenCV usually comes with this file built-in
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Error: Could not load face detector XML.")
        exit()
    return face_cascade

def preprocess_face(face_img):
    # 1. Convert BGR (OpenCV standard) to RGB (Model standard)
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to 64x64
    resized_face = cv2.resize(rgb_face, IMG_SIZE)
    
    # 3. Normalize pixel values (0-255 -> 0-1)
    # This is CRITICAL because we used rescale=1./255 in training
    normalized_face = resized_face / 255.0
    
    # 4. Expand dimensions to match batch shape: (1, 64, 64, 3)
    reshaped_face = np.expand_dims(normalized_face, axis=0)
    
    return reshaped_face

def main():
    # 1. Load Model
    print(f"🔄 Loading model: {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default camera
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    face_cascade = load_face_detector()
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    print("🚀 Starting Real-Time Stress Detection... Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for Face Detection (faster)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60) # Ignore tiny faces to save processing
        )

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Crop the face region
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess
                processed_face = preprocess_face(face_roi)
                
                # Predict
                predictions = model.predict(processed_face, verbose=0)
                class_index = np.argmax(predictions)
                confidence = np.max(predictions)
                
                label_text = f"{CLASS_LABELS[class_index]} ({confidence*100:.1f}%)"
                color = COLORS[class_index]

                # Draw Rectangle & Label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception as e:
                pass # Skip if face processing fails (e.g., face too close to edge)

        # --- Calculate and Display FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Result
        cv2.imshow('Stress Detection System', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  '''