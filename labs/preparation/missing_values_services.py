import pandas as pd

from helpers.dslabs_functions import mvi_by_dropping, mvi_by_filling
from labs.preparation.missing_values_functions import impute_credithistory, impute_column_finance


def init_impute(data):
    # Drops no variables and 4 records (0.004%)
    df = mvi_by_dropping(data, min_pct_per_variable=0.75, min_pct_per_record=0.85)
    df.reset_index(drop=True, inplace=True)

    return df


def imputate_missing_values_services(data, save=False):
    df = init_impute(data)

    # Custom imputation
    impute_column_finance(df, 'Age', 'mode')
    impute_column_finance(df, 'Occupation', 'mode')
    impute_column_finance(df, 'Monthly_Inhand_Salary', 'mode')
    impute_column_finance(df, 'Num_Bank_Accounts', 'mode')
    impute_column_finance(df, 'NumofLoan', 'mode')
    impute_column_finance(df, 'NumofDelayedPayment', 'mean')
    impute_column_finance(df, 'ChangedCreditLimit', 'mean')
    impute_column_finance(df, 'NumCreditInquiries', 'mode')
    impute_column_finance(df, 'CreditMix', 'mode')
    impute_credithistory(df)
    impute_column_finance(df, 'Amountinvestedmonthly', 'mean', rounding=5)
    impute_column_finance(df, 'Payment_Behaviour', 'mode')
    impute_column_finance(df, 'MonthlyBalance', 'mean', rounding=5)

    try:
        df.drop(columns=["Customer_ID"], inplace=True)
    except KeyError:
        pass

    if save:
        df.to_csv('../../datasets/prepared/class_credit_score_2_1.csv', index=False)

    return df


def impute_credit_score_knn(data, save=False):
    df = init_impute(data)
    df.drop(['Customer_ID'], axis=1, inplace=True)
    df = mvi_by_filling(df, 'knn')
    if save:
        df.to_csv('../../datasets/prepared/class_credit_score_2_knn.csv', index=False)

    return df


if __name__ == "__main__":
    fin = pd.read_csv('../../datasets/prepared/class_credit_score_encoded_1.csv', na_values="")
    # imputate_missing_values_services(fin, save=True)
    # impute_credit_score_knn(fin, save=True)
