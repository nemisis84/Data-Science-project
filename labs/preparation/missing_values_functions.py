import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import show

from helpers.dslabs_functions import mvi_by_dropping
from sklearn.impute import SimpleImputer, KNNImputer


def check_differing_values(fin, column):
    # Group by 'Customer_ID' and get unique values in 'column'
    # nan is counted as a value
    unique_values = fin.groupby('Customer_ID')[column].unique()

    # Filter the 'Customer_IDs' where the array has length 2 or more, meaning that there are differing values
    selected_customer_ids = unique_values[unique_values.apply(lambda x: len(x) > 2)].index

    print(f"There are {selected_customer_ids.size} customers with differing values for {column}")


def count_distribution_for_missing_values_per_customer(fin, column):
    counts = fin[fin[column].isna()]['Customer_ID'].value_counts()

    # Count how many times each count value occurs
    count_distribution = counts.value_counts().sort_index()

    print(count_distribution)


def impute_values(group, diff):
    start = group['Credit_History_Age'].dropna().iloc[0]
    return pd.Series(start + diff * np.arange(len(group)), index=group.index)


def impute_credithistory(df):
    diff = df['Credit_History_Age'][3] - df['Credit_History_Age'][2]
    df['Credit_History_Age'] = df.groupby('Customer_ID').apply(impute_values, diff=diff).reset_index(level=0, drop=True)


def impute_column(df, column_name, strategy='mean', rounding=0):
    if strategy == 'mean':
        mean_imputer = SimpleImputer(strategy='mean')
        df[column_name] = mean_imputer.fit_transform(df[[column_name]])
        df[column_name] = round(df[column_name], rounding)
    elif strategy == 'median':
        median_imputer = SimpleImputer(strategy='median')
        df[column_name] = median_imputer.fit_transform(df[[column_name]])
        df[column_name] = round(df[column_name], rounding)
    elif strategy == 'mode':
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df[column_name] = mode_imputer.fit_transform(df[[column_name]])
    else:
        print("Invalid strategy specified")


def impute_column_finance(df, column_name, method='mean', rounding=0):
    """
    Impute missing values in a column by either mean, mode or median. This groups together all rows with the same
    Customer_ID and uses that to impute the missing values.
    :param df: DataFrame
    :param column_name: String
    :param method: 'mean', 'mode' or 'median'
    :param rounding: int
    :return: None
    """
    if method == 'mean':
        mean_values = df.groupby('Customer_ID')[column_name].transform('mean')
        mean_values = round(mean_values, rounding)
        df[column_name] = df[column_name].fillna(mean_values)
    elif method == 'mode':
        mode_values = df.groupby('Customer_ID')[column_name].transform(lambda x: x.mode()[0])
        df[column_name] = df[column_name].fillna(mode_values)
    elif method == 'median':
        median_values = df.groupby('Customer_ID')[column_name].transform('median')
        median_values = round(median_values, rounding)
        df[column_name] = df[column_name].fillna(median_values)
    else:
        print("Invalid method specified")


def reasoning_services():
    """
    Finances dataset consists of a lot of records based on the same customer. This means that we can imputate missing
    values by grouping together all rows with the same Customer_ID and then imputating the missing values based on that.
    We therefore need to check how many missing values there are per customer, to see if we can actually get a good
    estimate of the missing values.
    """
    fin = pd.read_csv("../../datasets/prepared/class_credit_score_encoded_1.csv")
    # Should remove this line as it skews analysis (a bit)
    fin = mvi_by_dropping(fin, min_pct_per_variable=0.0, min_pct_per_record=0.9)  # Drops ~ 400 rows of 100k

    count_distribution_for_missing_values_per_customer(fin, "NumofDelayedPayment")
    '''
    Output:
    count
    1    4069
    2    1113
    3     156
    4      15  # <--- 15 customers have 4 missing values
    5       1  # <--- 1 customer has 5 missing values

    Based on this, we can imputate the missing values for the rest of the customers, but for these 16 it will be a bit
    misleading. Given that 16 << 12500 (0.1%), we can just take it for what it is
    '''

    count_distribution_for_missing_values_per_customer(fin, "ChangedCreditLimit")
    '''
    count
    1    1790
    2     108
    3       8
    No issues here, we can imputate the missing values
    '''

    count_distribution_for_missing_values_per_customer(fin, "NumCreditInquiries")
    '''
    count
    1    1656
    2     126
    3       4
    No issues here, we can imputate the missing values
    '''

    count_distribution_for_missing_values_per_customer(fin, "CreditMix")
    check_differing_values(fin, "CreditMix")
    '''
    count
    1    4276
    2    3597
    3    1865
    4     558
    5     110
    6      12

    Here we have a lot of missing values.
    CreditMix is a symbolic value:
    Bad - 0
    Standard - 1
    Good - 2

    After running check_differing_values, we can see that there are no customers who have more than one value for CreditMix.
    Meaning imputation by mode is fine here.
    '''

    count_distribution_for_missing_values_per_customer(fin, "Amountinvestedmonthly")
    '''
    count
    1    3168
    2     517
    3      42
    4       2
    5       2
    I hope using mean is fine here, but there are some outliers here (usually the exact amount 10 000)
    '''

    count_distribution_for_missing_values_per_customer(fin, "Payment_Behaviour")
    '''
    count
    1    4403
    2    1166
    3     194
    4      23
    5       2

    25 people are 0.2% of the total, so imputing here should be fine?
    '''

    count_distribution_for_missing_values_per_customer(fin, "MonthlyBalance")
    '''
    count
    1    755
    2    146
    3     25
    4     10
    5      1
    '''


if __name__ == "__main__":
    reasoning_services()
