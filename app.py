from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Načítaj model
model = joblib.load("email_filter_model.pkl")

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

@app.route("/", methods=["GET"])
def root():
    return "Email filter API is running 🚀"

if __name__ == "__main__":
    app.run(debug=True)
