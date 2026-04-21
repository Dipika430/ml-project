import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# STEP 1: Load Dataset
# =========================
iris = load_iris()
X = iris.data
y = iris.target


# =========================
# STEP 2: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# STEP 3: Preprocessing
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# STEP 4: Train Model
# =========================
model = LogisticRegression(max_iter=300, C=0.5)
model.fit(X_train, y_train)


# =========================
# STEP 5: Evaluate Model
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# =========================
# STEP 6: Save Model (SAFE PATH)
# =========================

# Get current file directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Go to parent folder → ml-project
project_root = os.path.abspath(os.path.join(base_dir, ".."))

# Create models folder if not exists
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)

# Define model path
model_path = os.path.join(model_dir, "model_v2.pkl")

# Save model
joblib.dump(model, model_path)

print("Model saved at:", model_path)