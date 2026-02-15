from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math_score = float(request.form["math_score"])
        reading_score = float(request.form["reading_score"])
        writing_score = float(request.form["writing_score"])

        X = np.array([[math_score, reading_score, writing_score]])
        pred = model.predict(X)[0]

        return render_template("index.html", prediction_text=f"Predicted Race/Ethnicity: {pred}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

