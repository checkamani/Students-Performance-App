import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (Kaggle file you already have: StudentsPerformance.csv)
df = pd.read_csv("StudentsPerformance.csv")

# Target: race/ethnicity
y = df["race/ethnicity"]

# Features: scores only (the assignment says from math, reading, writing)
X = df[["math score", "reading score", "writing score"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline (scaling + multinomial logistic regression)
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, multi_class="multinomial"))
    ]
)

clf.fit(X_train, y_train)

# Quick evaluation (prints accuracy in terminal)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")

# Save model bundle (simple here: just the pipeline)
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model trained and saved!")
