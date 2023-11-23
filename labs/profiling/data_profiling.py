import pandas as pd
import numpy as np

# Health domain – Pos covid

df = pd.read_csv('../../datasets/class_pos_covid.csv')
print("Classification pos covid")

# Records x variables
num_records = len(df)
num_variables = df.shape[1]
print(f"Number of Records: {num_records}")
print(f"Number of Variables: {num_variables}")

# Nr variables per type
variable_types = df.dtypes.value_counts()
print(variable_types)

# Find missing values
missing_values = df.isna().sum()
total_missing_values = missing_values.sum()
print(f"Total Missing Values: {total_missing_values}")

# Services domain - Credit score

df = pd.read_csv('../../datasets/class_credit_score.csv')
print("\nClassification credit score")

# Records x variables
num_records = len(df)
num_variables = df.shape[1]
print(f"Number of Records: {num_records}")
print(f"Number of Variables: {num_variables}")

# Nr variables per type
variable_types = df.dtypes.value_counts()
print(variable_types)

# Find missing values
missing_values = df.isna().sum()
total_missing_values = missing_values.sum()
print(f"Total Missing Values: {total_missing_values}")
