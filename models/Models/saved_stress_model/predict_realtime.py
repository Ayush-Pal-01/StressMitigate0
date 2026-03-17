import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import os

# ==========================================
# SETUP
# ==========================================
# Force CPU for inference to avoid "Out of Memory" errors on your laptop
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def load_prediction_model():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load the Tokenizer and Model
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully on CPU!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def predict_stress(text, tokenizer, model):
    # 1. Preprocess the text (Tokenize)
    inputs = tokenizer(
        text, 
        return_tensors="tf", 
        truncation=True, 
        padding=True, 
        max_length=128
    )

    # 2. Get Prediction (Logits)
    outputs = model(inputs)
    logits = outputs.logits

    # 3. Convert to Probabilities (Softmax)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    
    # 4. Get the winning class
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    # Map 0/1 to labels (Adjust based on your specific dataset labels)
    # Usually: 0 = No Stress / Neutral, 1 = Stress
    label = "STRESS DETECTED" if predicted_class == 1 else "No Stress"
    
    return label, confidence

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    tokenizer, model = load_prediction_model()

    if model:
        print("\n" + "="*40)
        print("🤖 AI STRESS DETECTOR READY")
        print("Type 'exit' to quit.")
        print("="*40 + "\n")

        while True:
            user_input = input("Enter a sentence: ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            label, score = predict_stress(user_input, tokenizer, model)
            
            # Print result with color
            color = "\033[91m" if label == "STRESS DETECTED" else "\033[92m" # Red for Stress, Green for No
            reset = "\033[0m"
            
            print(f"Prediction: {color}{label}{reset} (Confidence: {score*100:.2f}%)")
            print("-" * 30)
