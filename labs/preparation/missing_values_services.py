import pandas as pd

from helpers.dslabs_functions import mvi_by_dropping, mvi_by_filling
from labs.preparation.missing_values_functions import impute_credithistory, impute_column_finance


def imputate_missing_values_services():
    filename = '../../datasets/prepared/class_credit_score_encoded_1.csv'
    df = pd.read_csv(filename, na_values="")

    # Drop 400 rows with missing values, then reset indices
    # We should also test for dropping and not dropping records
    df = mvi_by_dropping(df, min_pct_per_variable=0.0, min_pct_per_record=0.9)
    df.reset_index(drop=True, inplace=True)

    # Custom imputation
    impute_column_finance(df, 'Age', 'mode')
    impute_column_finance(df, 'Occupation', 'mode')
    impute_column_finance(df, 'Monthly_Inhand_Salary', 'mode')
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

    df.to_csv('../../datasets/prepared/class_credit_score_2_1.csv', index=False)

    # KNN imputation
    df = pd.read_csv(filename, na_values="")
    mvi_by_filling(df, 'knn').to_csv('../../datasets/prepared/class_credit_score_2_knn.csv', index=False)


if __name__ == "__main__":
    imputate_missing_values_services()
