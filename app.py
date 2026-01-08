from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "StressMitigate Backend Running!"

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "")

    # Dummy stress logic (Week 1)
    if "exam" in text.lower() or "pressure" in text.lower():
        stress = "HIGH"
    else:
        stress = "LOW"

    return jsonify({
        "stress_level": stress,
        "message": "You are not alone. Take a slow breath."
    })

if __name__ == "__main__":
    app.run(debug=True)
