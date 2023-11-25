import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import show

from helpers.dslabs_functions import mvi_by_dropping


def check_differing_values(fin, column):
    # Group by 'Customer_ID' and get unique values in 'column'
    # nan is counted as a value
    unique_values = fin.groupby('Customer_ID')[column].unique()

    # Filter the 'Customer_IDs' where the array has length 2 or more, meaning that there are differing values
    selected_customer_ids = unique_values[unique_values.apply(lambda x: len(x) > 2)].index

    print(f"There are {selected_customer_ids.size} customers with differing values for {column}")


fin = pd.read_csv("../../datasets/prepared/class_credit_score_encoded_1.csv")


def mvi_amputation(fin):
    # find missing values
    mv = {}
    for col in fin.columns:
        num = fin[col].isna().sum()
        if num > 0:
            mv[col] = num

    # print(mv)
    print(fin.shape)

    print(f"col\trow")
    for i in range(101):
        print({i / 100})
        print(mvi_by_dropping(fin, min_pct_per_variable=0.0, min_pct_per_record=(i / 100)).shape[0])


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


def impute_column(df, column_name, method='mean', rounding=0):
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


def render_mv(df):
    print(f"Dataset nr records={df.shape[0]}", f"nr variables={df.shape[1]}")

    mv: dict[str, int] = {}
    for var in df.columns:
        nr: int = df[var].isna().sum()
        if "Loan" not in var and "Not Specified" not in var and "cos" not in var and "sin" not in var:
            mv[var] = nr

    print(f"Missing values: {mv}")
    for i in mv:
        print(f"{i}\t{mv[i]}")

    variables = list(mv.keys())
    missing_values = list(mv.values())
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.figure(figsize=(20, 8))
    bars = sns.barplot(x=variables, y=missing_values)
    plt.xticks(rotation=45, ha='right')

    plt.title('Number of Missing Values per Variable', fontsize=24)
    plt.ylabel('Number of Missing Values', fontsize=18)

    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), ',.0f'),  # No decimal places, include commas as thousands separators.
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='center',
                      size=13, xytext=(0, 8),
                      textcoords='offset points')

    # sns.despine()
    plt.tight_layout()
    # plt.savefig('../../figures/temp/missing_values_per_variable.png')
    show()


# Reasonings for the imputation modes finances
if __name__ == '__main__':
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
