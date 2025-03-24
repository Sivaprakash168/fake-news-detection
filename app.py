from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ Model and Vectorizer loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or vectorizer file not found. Train the model first.")
    model, vectorizer = None, None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, vectorizer = None, None

# Serve Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Serve About Page
@app.route("/about")
def about():
    return render_template("about.html")

# API Endpoint for Fake News Detection
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500

    try:
        data = request.get_json()
        news_text = data.get("news", "").strip()

        if not news_text:
            return jsonify({"error": "No news text provided"}), 400

        # Process input text and make prediction
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]

        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
