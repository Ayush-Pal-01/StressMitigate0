from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def detect_stress(text):
    result = sentiment_analyzer(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "NEGATIVE":
        return "HIGH", score
    elif label == "POSITIVE":
        return "LOW", score
    else:
        return "MEDIUM", score
