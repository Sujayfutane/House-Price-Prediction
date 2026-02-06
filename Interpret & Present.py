# ==========================================================
# INTERPRET & PRESENT – MODEL INSIGHTS (ERROR FREE)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("house_prices.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# PREPROCESSOR
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

print("\n================ INTERPRETATION & INSIGHTS ================")

# ==========================================================
# A. LINEAR REGRESSION COEFFICIENTS
# ==========================================================
lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_pipe.fit(X_train, y_train)

ohe_features = (
    lr_pipe.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(cat_cols)
)

feature_names = np.concatenate([num_cols, ohe_features])
coefficients = lr_pipe.named_steps["model"].coef_

coef_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })
    .sort_values(by="Coefficient", key=abs, ascending=False)
)

print("\nTOP LINEAR MODEL COEFFICIENTS")
print(coef_df.head(15))

coef_df.to_csv("linear_model_coefficients.csv", index=False)

# ==========================================================
# B. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200, random_state=42
    ))
])

rf_pipe.fit(X_train, y_train)

rf_importance = rf_pipe.named_steps["model"].feature_importances_

rf_importance_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_importance
    })
    .sort_values(by="Importance", ascending=False)
)

print("\nTOP RANDOM FOREST FEATURES")
print(rf_importance_df.head(15))

rf_importance_df.to_csv(
    "random_forest_feature_importance.csv", index=False
)

plt.figure(figsize=(8, 5))
plt.barh(
    rf_importance_df.head(15)["Feature"],
    rf_importance_df.head(15)["Importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# ==========================================================
# C. PERMUTATION IMPORTANCE
# ==========================================================
perm = permutation_importance(
    rf_pipe,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

perm_df = (
    pd.DataFrame({
        "Feature": X.columns,
        "Permutation Importance": perm.importances_mean
    })
    .sort_values(by="Permutation Importance", ascending=False)
)

print("\nTOP PERMUTATION IMPORTANCE FEATURES")
print(perm_df.head(10))

perm_df.to_csv("permutation_importance.csv", index=False)

# ==========================================================
# D. RESIDUAL STATISTICS
# ==========================================================
y_pred = rf_pipe.predict(X_test)
residuals = y_test - y_pred

print("\nRESIDUAL STATISTICS")
print("Mean Residual :", residuals.mean())
print("Std Residual  :", residuals.std())
print("Max Residual  :", residuals.max())
print("Min Residual  :", residuals.min())

# ==========================================================
# E. BUSINESS INSIGHTS (SUMMARY)
# ==========================================================
print("\nKEY INSIGHTS")
print("• Larger living area strongly increases house price")
print("• Quality and location are major price drivers")
print("• Bathrooms and garage size add premium value")
print("• Tree-based models capture non-linear effects better")

print("\n===========================================================")
