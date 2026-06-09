from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model + feature order
model, feature_names = pickle.load(open("loan_fraud_model.pkl", "rb"))

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- PREDICTION PAGE ----------------
@app.route("/predict_page")
def predict_page():
    return render_template("index.html")

# ---------------- PREDICT FUNCTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    input_values = []

    # Collect inputs safely
    for feature in feature_names:
        value = request.form.get(feature)

        if value is None or value == "":
            value = 0

        input_values.append(float(value))

    final_input = [input_values]

    # Prediction
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    # Fraud or Not
    if prediction == 1:
        result_text = "⚠ Fraud Detected"
        css_class = "result-fraud"
    else:
        result_text = "✅ No Fraud Detected"
        css_class = "result-safe"

    return render_template(
        "result.html",
        result_text=result_text,
        probability=round(probability * 100, 2),
        css_class=css_class
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)