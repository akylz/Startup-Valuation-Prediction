import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

RAW_DATA_PATH = "../data/raw/startup_growth_investment_data.csv"
PROCESSED_DATA_PATH = "../data/processed/preprocessed_data.csv"

def load_data():
    return pd.read_csv(RAW_DATA_PATH)

def preprocess_data(df):
    df["Startup Age"] = 2025 - df["Year Founded"]
    df["Investment-to-Valuation Ratio"] = df["Investment Amount (USD)"] / df["Valuation (USD)"]
    df["Funding Rounds per Investor"] = df["Funding Rounds"] / df["Number of Investors"]
    df["Growth-Adjusted Investment"] = df["Investment Amount (USD)"] * df["Growth Rate (%)"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    label_encoders = {}
    for col in ["Industry", "Country"]:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    features = ["Funding Rounds", "Investment Amount (USD)", "Number of Investors",
                "Growth Rate (%)", "Startup Age", "Industry", "Country",
                "Investment-to-Valuation Ratio", "Funding Rounds per Investor",
                "Growth-Adjusted Investment"]
    
    X = df[features]
    y = df["Valuation (USD)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=features)
    processed_df["Valuation (USD)"] = y

    return processed_df

def save_data(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_data()
    processed_df = preprocess_data(df)
    save_data(processed_df)
    print("Preprocessing complete!")
