# =====================================
# DATA ANALYSIS: LOAD, CHECK, VISUALIZE
# =====================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Replace 'data.csv' with your dataset path
df = pd.read_csv('house_prices.csv')

print("Dataset Loaded Successfully!")
print(df.head())

# -------------------------------
# 2. Basic Dataset Information
# -------------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# 3. Check Missing Values
# -------------------------------
print("\nMissing Values Count:")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# -------------------------------
# 4. Visualize Relationships
# -------------------------------

# Pairplot (numerical relationships)
sns.pairplot(df)
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Example: Scatter plot (change column names)
# sns.scatterplot(x='column1', y='column2', data=df)
# plt.show()
