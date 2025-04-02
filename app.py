from flask import Flask, request, jsonify
import joblib
import os

# Načítaj model
model = joblib.load("email_filter_model.pkl")

# Inicializuj Flask aplikáciu
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return "Email filter API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Predikcia pravdepodobností
    probs = model.predict_proba([text])[0]
    response = {
        "prob_not_relevant": float(probs[0]),
        "prob_relevant": float(probs[1])
    }
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
