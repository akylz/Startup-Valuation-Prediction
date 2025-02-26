import pandas as pd
import joblib

MODEL_PATH = "../models/random_forest.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def get_user_input():
    print("\nEnter startup details for valuation prediction:\n")
    
    funding_rounds = int(input("Number of funding rounds: "))
    investment_amount = float(input("Total investment amount (USD): "))
    num_investors = int(input("Total number of investors: "))
    growth_rate = float(input("Growth rate (%): "))
    startup_age = int(input("Startup age (years): "))
    industry = int(input("Industry (encoded as an integer): "))
    country = int(input("Country (encoded as an integer): "))

    inv_to_valuation_ratio = investment_amount / 1e10  # Normalized assumption
    funding_rounds_per_investor = funding_rounds / num_investors if num_investors else 0
    growth_adj_investment = investment_amount * growth_rate

    return pd.DataFrame([[
        funding_rounds, investment_amount, num_investors, growth_rate,
        startup_age, industry, country, inv_to_valuation_ratio, 
        funding_rounds_per_investor, growth_adj_investment
    ]], columns=[
        "Funding Rounds", "Investment Amount (USD)", "Number of Investors",
        "Growth Rate (%)", "Startup Age", "Industry", "Country",
        "Investment-to-Valuation Ratio", "Funding Rounds per Investor",
        "Growth-Adjusted Investment"
    ])

def predict_new_data(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    user_data = get_user_input()
    prediction = predict_new_data(user_data)
    print(f"\nPredicted Valuation: {prediction[0]:,.2f} USD\n")
