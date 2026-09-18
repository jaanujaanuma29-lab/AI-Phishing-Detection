from flask import Flask, render_template, request
import pandas as pd
import joblib
from feature_extraction import extract_features

app = Flask(__name__)

model = joblib.load("phishing_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    risk_score = None
    risk_level = None
    features = None
    url = ""

    if request.method == "POST":

        url = request.form["url"]

        features = extract_features(url)

        input_data = pd.DataFrame([features])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            result = "PHISHING URL"
        else:
            result = "LEGITIMATE URL"

        risk_score = round(probability[1] * 100, 2)

        if risk_score <= 30:
            risk_level = "LOW RISK"
        elif risk_score <= 70:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "HIGH RISK"

    return render_template(
        "index.html",
        result=result,
        risk_score=risk_score,
        risk_level=risk_level,
        features=features,
        url=url
    )


if __name__ == "__main__":
    app.run(debug=True)