from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved bundle: {"model": model, "columns": [...]}
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
columns = bundle["columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values by name (must match index.html)
        reading = float(request.form["reading_score"])
        writing = float(request.form["writing_score"])

        # Build full feature row (zeros + set our two numeric inputs)
        x = {col: 0 for col in columns}
        x["reading score"] = reading
        x["writing score"] = writing

        row = np.array([x[col] for col in columns]).reshape(1, -1)

        pred = model.predict(row)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Math Score: {pred:.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
