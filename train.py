import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("StudentsPerformance.csv")

df_enc = pd.get_dummies(df, drop_first=True)

y = df_enc["math score"]
X = df_enc.drop("math score", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

bundle = {"model": model, "columns": list(X.columns)}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Model trained and saved!")
