import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

df = pd.read_csv("../data/processed/preprocessed_data.csv")

X = df.drop(columns=["Valuation (USD)"])
y = df["Valuation (USD)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.7, random_state=42)
}

best_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    best_models[name] = model

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name}: MAE = {mae:.2f}, R² = {r2:.2f}")

    os.makedirs("../models", exist_ok=True)
    model_path = f"../models/{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
