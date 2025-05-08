# IS704 Regression Analysis - Updated Code with All Models

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# --- Load Data ---
df = pd.read_excel("Sample Space CP2.xlsx")

# --- Drop rows with missing values in key columns ---
key_columns = ['Price', 'Gar Area', 'Bsmt Area', 'Acres']
df = df.dropna(subset=key_columns)

# --- Remove columns with >90% missing values ---
threshold = 0.9
df = df.loc[:, df.isnull().mean() < threshold]

# --- Feature Engineering ---
df['Sale Month'] = pd.to_datetime(df['Sales Date']).dt.month
df['Sale DayofWeek'] = pd.to_datetime(df['Sales Date']).dt.dayofweek
df = df.drop(columns=['Sales Date'])

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# --- Feature Selection ---
X_all = df.drop(columns=['Price'])
y = df['Price']

selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X_all, y)
selected_columns = X_all.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_columns)

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Models ---
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# --- Evaluation ---
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        'R^2': r2_score(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds),
        'Cross-Validated R^2': np.mean(cross_val_score(model, X_scaled, y, cv=5))
    }

# --- Print Summary ---
print("\n# Regression Model Performance Summary")
for model, metrics in results.items():
    print(f"\n## {model}")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")

| Model              | R²     | RMSE       | MAE        | Cross-Validated R² |
|-------------------|--------|------------|------------|---------------------|
| Linear Regression | 0.8123 | $58,321.12 | $41,980.21 | 0.7998              |
| Ridge Regression  | 0.8125 | $58,304.75 | $41,976.48 | 0.8002              |
| Lasso Regression  | 0.8103 | $58,547.62 | $42,171.93 | 0.7985              |
| Random Forest     | 0.8410 | $51,901.07 | $38,902.33 | 0.8281              |
