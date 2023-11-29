import re
from math import pi, sin, cos

import numpy as np
import pandas as pd


def expand_loans(row, all_loans):
    loans = row['Type_of_Loan'].split(',')
    return [1 if loan in loans else 0 for loan in all_loans]


def clean_loan_type(loan_type):
    return loan_type.strip().replace('and ', '')


def handle_loans(df):
    all_loans = set(clean_loan_type(loan)
                    for loan_list in df['Type_of_Loan'].dropna()
                    for loan in loan_list.split(','))

    expanded_rows = []
    for loan_list in df['Type_of_Loan']:
        loans = loan_list.split(',') if pd.notna(loan_list) else []
        cleaned_loans = [clean_loan_type(loan) for loan in loans]
        row = [1 if loan in cleaned_loans else 0 for loan in all_loans]
        expanded_rows.append(row)
    expanded_df = pd.DataFrame(expanded_rows, columns=list(all_loans))
    df = pd.concat([df.reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)
    return df


def transform_bools(df, keyword):
    for col in df.columns:
        if col.startswith(keyword):
            df[col] = df[col].astype(int)
    return df


def clean_and_convert(s):
    if pd.isna(s) or s == "nan":
        return np.nan

    try:
        num = int(s)  # Check if s is a number
        if num < 0 or num > 122:  # Check if the age is reasonable
            return np.nan
    except ValueError:  #
        s = re.sub(r'[^0-9.]+', '', s)  # Remove non-numeric characters
        if int(s) < 0 or int(s) > 122:
            return np.nan

    return int(s) if s.isdigit() else float(s)


def handle_months(df):
    month_encoding = {
        "January": 0,
        "February": pi / 6,
        "March": 2 * pi / 6,
        "April": 3 * pi / 6,
        "May": 4 * pi / 6,
        "June": 5 * pi / 6,
        "July": 6 * pi / 6,
        "August": 7 * pi / 6,
    }

    encoding = {
        "Month": month_encoding,
    }

    df = df.replace(encoding)
    df = encode_cyclic_variables(df, ["Month"])

    return df


def encode_cyclic_variables(df, vars):
    for v in vars:
        x_max: float | int = max(df[v])
        df[v + "_sin"] = df[v].apply(lambda x: round(sin(2 * pi * x / x_max), 3))
        df[v + "_cos"] = df[v].apply(lambda x: round(cos(2 * pi * x / x_max), 3))
    return df


def calulate_age(age_categories):
    mean_ages = []
    for value in age_categories:
        split = value.split()
        years, months = int(split[0]), int(split[3])
        mean_ages.append((years + months / 12))
    return mean_ages


def remove_negatives(df, exclude_cols=None):
    if exclude_cols is None:
        # If no specific columns provided, apply to all columns
        mask = df < 0
    else:
        # Apply the mask only to the specified columns
        include_cols = df.columns.drop(exclude_cols)
        mask = df[include_cols] < 0

    # Replace negative values with NaN
    df[mask] = np.nan
    return df


def encode_services(df):
    df.drop(columns=["ID", "SSN", "Name"], inplace=True)

    df = handle_loans(df)

    credit_mix_encoding = {"Good": 2, "Standard": 1, "Bad": 0, "nan": np.nan}
    payment_of_min_amount_encoding = {"No": 0, "NM": 1, "Yes": 2, "nan": np.nan}
    payment_behaviour_encoding = {'High_spent_Small_value_payments': 5,
                                  'Low_spent_Large_value_payments': 0,
                                  'Low_spent_Medium_value_payments': 1,
                                  'Low_spent_Small_value_payments': 2,
                                  'High_spent_Medium_value_payments': 4,
                                  "nan": np.nan,
                                  'High_spent_Large_value_payments': 3}
    credit_score_encoding = {"Good": 1, "Poor": 0}
    occupation_encoding = {'Scientist': 0, "nan": np.nan, 'Teacher': 12, 'Engineer': 1, 'Entrepreneur': 8,
                           'Developer': 3, 'Lawyer': 7, 'Media_Manager': 10, 'Doctor': 4, 'Journalist': 11,
                           'Manager': 5, 'Accountant': 6, 'Musician': 14, 'Mechanic': 9, 'Writer': 13,
                           'Architect': 2}

    age_value_counts = df["Credit_History_Age"].value_counts()
    mean_ages = calulate_age(age_value_counts.index)
    credit_history_age_encoding = {age_value_counts.index[i]: mean_ages[i] for i in range(len(mean_ages))}
    credit_history_age_encoding["nan"] = np.nan

    encoding = {}
    encoding["CreditMix"] = credit_mix_encoding
    encoding["Payment_of_Min_Amount"] = payment_of_min_amount_encoding
    encoding["Payment_Behaviour"] = payment_behaviour_encoding
    encoding["Credit_Score"] = credit_score_encoding
    encoding["Credit_History_Age"] = credit_history_age_encoding
    encoding["Occupation"] = occupation_encoding
    df = df.replace(encoding, inplace=False)

    df['Age'] = df['Age'].apply(clean_and_convert)

    df = handle_months(df)
    df.drop(columns=["Month", "Type_of_Loan"], inplace=True)

    df = remove_negatives(df, exclude_cols=["Customer_ID", "Month_cos", "Month_sin", "ChangedCreditLimit",
                                            "Delay_from_due_date"])

    df.to_csv('../../datasets/prepared/1_Credit_Score.csv', index=False)
    return df
