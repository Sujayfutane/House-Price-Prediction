# =====================================
# LINEAR REGRESSION FROM SCRATCH
# USING PROVIDED HOUSE PRICES DATASET
# =====================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("house_prices.csv")

print("Dataset Loaded")
print(df.head())

# -------------------------------
# 2. Basic Preprocessing (Required)
# -------------------------------
# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 3. Features & Target
# -------------------------------
# Target column as per your dataset
X = df.drop("Price", axis=1)
y = df["Price"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Feature Scaling (Important)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# -------------------------------
# 6. Linear Regression from Scratch
# -------------------------------
m, n = X_train.shape

weights = np.zeros((n, 1))
bias = 0

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):

    # Hypothesis: ŷ = Xw + b
    y_pred = np.dot(X_train, weights) + bias

    # Cost Function (MSE)
    cost = (1 / m) * np.sum((y_pred - y_train) ** 2)

    # Gradients
    dw = (2 / m) * np.dot(X_train.T, (y_pred - y_train))
    db = (2 / m) * np.sum(y_pred - y_train)

    # Update weights & bias
    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {cost:.2f}")

# -------------------------------
# 7. Model Evaluation
# -------------------------------
y_test_pred = np.dot(X_test, weights) + bias

rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))
print("\nRMSE:", rmse)

print("\nTraining Completed Successfully ✅")
