
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading diabetes dataset...")

df = pd.read_csv("dataset/pima-indians-diabetes.csv", header=None)
df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]
# Separate input features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model, "artifacts/diabetes_model.joblib")
joblib.dump(list(X.columns), "artifacts/feature_names.joblib")

print("\nModel saved successfully!")
print("TRAINING COMPLETED SUCCESSFULLY!")
