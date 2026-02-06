# =====================================
# ADVANCED REGRESSION & MODEL COMPARISON
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("house_prices.csv")

# -------------------------------
# 2. Preprocessing
# -------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("Price", axis=1)
y = df["Price"]

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 3. LINEAR REGRESSION (BASELINE)
# =====================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# =====================================================
# 4. GRIDSEARCHCV (RIDGE & LASSO)
# =====================================================
ridge_params = {"alpha": [0.01, 0.1, 1, 10, 100]}
ridge_gs = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="neg_root_mean_squared_error")
ridge_gs.fit(X_train, y_train)

lasso_params = {"alpha": [0.01, 0.1, 1, 10]}
lasso_gs = GridSearchCV(Lasso(max_iter=5000), lasso_params, cv=5, scoring="neg_root_mean_squared_error")
lasso_gs.fit(X_train, y_train)

best_ridge = ridge_gs.best_estimator_
best_lasso = lasso_gs.best_estimator_

# =====================================================
# 5. RANDOM FOREST REGRESSION
# =====================================================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# =====================================================
# 6. XGBOOST (SAFE OPTIONAL)
# =====================================================
try:
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

except ImportError:
    rmse_xgb = None
    r2_xgb = None
    print("\nXGBoost not installed (skipping)")

# =====================================================
# 7. FEATURE IMPORTANCE (RANDOM FOREST)
# =====================================================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# =====================================================
# 8. RESIDUAL PLOT (LINEAR REGRESSION)
# =====================================================
residuals = y_test - y_pred_lr

plt.figure()
plt.scatter(y_pred_lr, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.show()

# =====================================================
# 9. ACTUAL vs PREDICTED PLOT
# =====================================================
plt.figure()
plt.scatter(y_test, y_pred_lr)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()

# =====================================================
# 10. FINAL RESULTS
# =====================================================
print("\nFINAL MODEL COMPARISON")
print("--------------------------------")
print(f"Linear Regression  | RMSE: {rmse_lr:.2f} | R²: {r2_lr:.2f}")
print(f"Best Ridge (CV)    | Alpha: {ridge_gs.best_params_}")
print(f"Best Lasso (CV)    | Alpha: {lasso_gs.best_params_}")
print(f"Random Forest     | RMSE: {rmse_rf:.2f} | R²: {r2_rf:.2f}")

if rmse_xgb:
    print(f"XGBoost            | RMSE: {rmse_xgb:.2f} | R²: {r2_xgb:.2f}")
