import pandas as pd
import joblib
from feature_extraction import extract_features

# Load trained model
model = joblib.load("phishing_model.pkl")

# Get URL from user
url = input("Enter a URL: ")

# Extract features
features = extract_features(url)

# Convert features into DataFrame
input_data = pd.DataFrame([features])

# Make prediction
prediction = model.predict(input_data)[0]

# Display result
print("\nPrediction:")

if prediction == 1:
    print("🔴 PHISHING URL")
else:
    print("🟢 LEGITIMATE URL")