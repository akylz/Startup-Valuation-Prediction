# Startup Valuation Prediction using Machine Learning

## Project Overview
This project predicts startup valuation based on investment amount, funding rounds, growth rate, and industry trends using machine learning models. The models were trained on a dataset containing startup financial information.

## Features
- **Data Preprocessing**: Handled missing values, encoded categorical features, and scaled numerical data.
- **Feature Engineering**: Created meaningful metrics like Investment-to-Valuation Ratio and Growth-Adjusted Investment.
- **Model Training & Optimization**:
  - Linear Regression
  - Random Forest Regressor (Best Performing Model)
  - Gradient Boosting Regressor
- **Hyperparameter Tuning**: Used GridSearchCV to optimize models for better accuracy.
- **Interactive Prediction**: Allows users to enter startup details and receive a valuation prediction.

## Model Performance
| Model               | MAE (Lower is Better) | R² Score (Higher is Better) |
|---------------------|----------------------|----------------------------|
| **Linear Regression** | **1.26B** | **0.91** |
| **Random Forest**    | **74.4M** | **1.00** |
| **Gradient Boosting** | **79.5M** | **1.00** |

## Technologies Used
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Joblib (for model saving)

## Project Structure
```
startup-valuation-predictor/
|-- data/            
|-- notebooks/        
|-- src/             
|-- models/          
|-- reports/         
|-- requirements.txt  
|-- README.md        
|-- main.py          
```

## How to Use
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing
```bash
python src/preprocessing.py
```

### 3. Train Models
```bash
python src/train.py
```

### 4. Make Predictions
Run the interactive prediction script:
```bash
python src/predict.py
```

## Next Steps
- Improve feature engineering by considering more financial metrics.
- Deploy the model as a web application using FastAPI or Streamlit.
- Experiment with Deep Learning models (Neural Networks).

---

**Date:** [2025, February]

