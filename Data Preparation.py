# =====================================
# DATA PREPROCESSING PIPELINE (FINAL)
# =====================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("house_prices.csv")

print("Initial Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# -------------------------------
# 3. Convert Categorical Variables
# -------------------------------
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCategorical Variables Converted Successfully")

# -------------------------------
# 4. Define Features & Target
# -------------------------------
target_column = 'Price'   # ✅ Correct target for your dataset

X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
