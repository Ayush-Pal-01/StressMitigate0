from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from models.text_model import detect_stress

app = FastAPI(title="StressMitigate API")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze_text")
def analyze_text(data: TextInput):
    stress_level, confidence = detect_stress(data.text)

    return {
        "stress_level": stress_level,
        "confidence": round(confidence, 2),
        "message": generate_response(stress_level)
    }

def generate_response(stress):
    if stress == "HIGH":
        return "I sense high stress. Let's pause and breathe slowly."
    elif stress == "MEDIUM":
        return "You seem a bit tense. A short break could help."
    else:
        return "You seem calm. Keep maintaining this balance."
