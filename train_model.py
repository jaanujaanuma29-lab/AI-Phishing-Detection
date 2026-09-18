import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/phishing_features.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

# Remove non-numeric / identifier columns
X = data.drop(columns=["url", "label", "tld"])

# Target
y = data["label"]


# ==========================================
# 3. CHECK CLASS DISTRIBUTION
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========")

print(y.value_counts())

print("\nClass percentages:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ==========================================
# 4. STRATIFIED TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# Probability of phishing class (1)
probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


# ==========================================
# 9. DISPLAY METRICS
# ==========================================

print("\n========== MODEL EVALUATION ==========")

print("Accuracy          :", round(accuracy * 100, 2), "%")
print("Precision         :", round(precision * 100, 2), "%")
print("Recall            :", round(recall * 100, 2), "%")
print("F1 Score          :", round(f1 * 100, 2), "%")
print("Balanced Accuracy :", round(balanced_accuracy * 100, 2), "%")
print("ROC-AUC           :", round(roc_auc, 4))
print("PR-AUC            :", round(pr_auc, 4))


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)

print("\nFormat:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 12. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n========== FEATURE IMPORTANCE ==========")

for index, row in feature_importance.iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


# ==========================================
# 13. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "phishing_model.pkl"
)

print("\nModel saved as phishing_model.pkl")

print("\n========== TRAINING COMPLETED ==========")