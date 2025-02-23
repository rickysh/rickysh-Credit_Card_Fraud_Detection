# Phase 1: Data Collection and Preprocessing (preprocessing.py)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# 1. Basic Info and Initial Exploration
print(df.info())
print(df.head())

# 2. Check for Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 3. Analyze Class Distribution
print("\nClass Distribution:")
print(df['Class'].value_counts(normalize=True))

# 4. Feature Scaling (Standardize 'Amount' and 'Time')
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop Original 'Amount' and 'Time'
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# 5. Save the Preprocessed Data
preprocessed_file_path = 'preprocessed_creditcard.csv'
df.to_csv(preprocessed_file_path, index=False)
print(f"Preprocessed dataset saved to {preprocessed_file_path}")
