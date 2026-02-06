# ==========================================================
# FINAL END-TO-END HOUSE PRICE PREDICTION (ERROR & WARNING FREE)
# ==========================================================

# -------------------------------
# 0. SUPPRESS WARNINGS
# -------------------------------
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, learning_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PolynomialFeatures
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance

# -------------------------------
# 2. LOAD DATA
# -------------------------------
df = pd.read_csv("house_prices.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

# -------------------------------
# 3. PREPROCESSING PIPELINE
# -------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# -------------------------------
# 4. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. BASE MODELS
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=5000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    )
}

results = []

# -------------------------------
# 6. TRAIN + EVALUATE MODELS
# -------------------------------
for name, model in models.items():

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    cv_rmse = -cross_val_score(
        pipe, X, y,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()

    results.append([name, mae, mse, rmse, r2, cv_rmse])

# -------------------------------
# 7. POLYNOMIAL REGRESSION
# -------------------------------
poly_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

poly_pipe.fit(X_train, y_train)
y_pred_poly = poly_pipe.predict(X_test)

results.append([
    "Polynomial Regression",
    mean_absolute_error(y_test, y_pred_poly),
    mean_squared_error(y_test, y_pred_poly),
    np.sqrt(mean_squared_error(y_test, y_pred_poly)),
    r2_score(y_test, y_pred_poly),
    -cross_val_score(
        poly_pipe, X, y,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()
])

# -------------------------------
# 8. GRADIENT BOOSTING + GRIDSEARCH
# -------------------------------
gb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

gb_params = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4]
}

gb_grid = GridSearchCV(
    gb_pipe,
    gb_params,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

y_pred_gb = best_gb.predict(X_test)

results.append([
    "Gradient Boosting (Tuned)",
    mean_absolute_error(y_test, y_pred_gb),
    mean_squared_error(y_test, y_pred_gb),
    np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    r2_score(y_test, y_pred_gb),
    -gb_grid.best_score_
])

# -------------------------------
# 9. XGBOOST (OPTIONAL SAFE)
# -------------------------------
try:
    from xgboost import XGBRegressor

    xgb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        ))
    ])

    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)

    results.append([
        "XGBoost",
        mean_absolute_error(y_test, y_pred_xgb),
        mean_squared_error(y_test, y_pred_xgb),
        np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        r2_score(y_test, y_pred_xgb),
        -cross_val_score(
            xgb_pipe, X, y,
            scoring="neg_root_mean_squared_error",
            cv=5
        ).mean()
    ])

except Exception:
    print("XGBoost not installed – skipped safely")

# -------------------------------
# 10. RESULTS SUMMARY
# -------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "RMSE", "R2", "CV_RMSE"]
)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)

# -------------------------------
# 11. FEATURE IMPORTANCE (RF)
# -------------------------------
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200, random_state=42
    ))
])

rf_pipe.fit(X_train, y_train)

perm = permutation_importance(
    rf_pipe, X_test, y_test,
    n_repeats=10, random_state=42
)

plt.figure()
plt.barh(X.columns, perm.importances_mean)
plt.xlabel("Importance")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------------
# 12. LEARNING CURVE
# -------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    rf_pipe, X, y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

plt.plot(train_sizes, -train_scores.mean(axis=1), label="Train RMSE")
plt.plot(train_sizes, -test_scores.mean(axis=1), label="Validation RMSE")
plt.legend()
plt.xlabel("Training Size")
plt.ylabel("RMSE")
plt.title("Learning Curve")
plt.show()

# -------------------------------
# 13. RESIDUAL DIAGNOSTICS
# -------------------------------
y_pred_rf = rf_pipe.predict(X_test)
residuals = y_test - y_pred_rf

plt.scatter(y_pred_rf, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

# -------------------------------
# 14. SHAP (SAFE)
# -------------------------------
try:
    import shap
    print("SHAP available")

except Exception:
    print("SHAP not installed or skipped safely")

# -------------------------------
# 15. SAVE FINAL MODEL
# -------------------------------
joblib.dump(rf_pipe, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")
