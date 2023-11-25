import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure, show
from helpers.dslabs_functions import plot_bar_chart, get_variable_types, define_grid, HEIGHT, plot_multibar_chart, \
    determine_outlier_thresholds_for_var, count_outliers, derive_date_variables, analyse_date_granularity, \
    mvi_by_filling, mvi_by_dropping
import seaborn as sns
import numpy as np


def check_occupation_row(row, df):
    if pd.isna(row['Occupation']):
        # Check row below
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
            if pd.notna(df['Occupation'].iloc[row.name + 1]):
                df['Occupation'].iloc[row.name] = df['Occupation'].iloc[row.name + 1]
                # Propagate the value to the rows above as well
                for i in range(1, 3):
                    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - i]:
                        if pd.isna(
                                df['Occupation'].iloc[row.name - i]
                        ):
                            df['Occupation'].iloc[row.name - i] = df['Occupation'].iloc[row.name + 1]
                        else:
                            break

                return
        # Check row above
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
            if pd.notna(df['Occupation'].iloc[row.name - 1]):
                df['Occupation'].iloc[row.name] = df['Occupation'].iloc[row.name - 1]
                return


def check_age_row(row, df):
    age = row['Age']

    if pd.isna(age) or age > 122:
        # Check row below
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
            age = df['Age'].iloc[row.name + 1]
            if pd.notna(age):
                if age < 122:
                    df['Age'].iloc[row.name] = age
                    # Propagate the value to the row above as well, unless it is the same age - 1
                    # (then the person has had a birthday)
                    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1] and age != df['Age'].iloc[
                        row.name - 1] + 1:
                        df['Age'].iloc[row.name - 1] = age
                    return

        # Check row above
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
            age = df['Age'].iloc[row.name - 1]
            if pd.notna(age):
                if age < 122:
                    df['Age'].iloc[row.name] = age
                    return


# We assume that people are not unemployed. !!!
def check_monthly_inhand_salary_row(row, df):
    monthly_inhand_salary = row['Monthly_Inhand_Salary']

    if pd.isna(monthly_inhand_salary):
        # Check row below
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
            monthly_inhand_salary = df['Monthly_Inhand_Salary'].iloc[row.name + 1]
            if pd.notna(monthly_inhand_salary):
                df['Monthly_Inhand_Salary'].iloc[row.name] = monthly_inhand_salary
                # Propagate the value to the rows above as well
                for i in range(1, 4):
                    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - i]:
                        if pd.isna(
                                df['Monthly_Inhand_Salary'].iloc[row.name - i]
                        ):
                            df['Monthly_Inhand_Salary'].iloc[row.name - i] = monthly_inhand_salary
                        else:
                            break
                return

        # Check row above
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
            monthly_inhand_salary = df['Monthly_Inhand_Salary'].iloc[row.name - 1]
            if pd.notna(monthly_inhand_salary):
                df['Monthly_Inhand_Salary'].iloc[row.name] = monthly_inhand_salary
                return


def impute_column(df, column_name, method='mean', rounding=True):
    """
    Impute missing values in a column by either mean, mode or median. This groups together all rows with the same
    Customer_ID and uses that to impute the missing values.
    :param df: DataFrame
    :param column_name: 'mean', 'mode' or 'median
    :param method: 'mean', 'mode' or 'median
    :param rounding: Boolean
    :return: None
    """
    if method == 'mean':
        mean_values = df.groupby('Customer_ID')[column_name].transform('mean')
        if rounding:
            mean_values = mean_values.round()
        df[column_name] = df[column_name].fillna(mean_values)
    elif method == 'mode':
        mode_values = df.groupby('Customer_ID')[column_name].transform(lambda x: x.mode()[0])
        df[column_name] = df[column_name].fillna(mode_values)
    elif method == 'median':
        median_values = df.groupby('Customer_ID')[column_name].transform('median')
        if rounding:
            median_values = median_values.round()
        df[column_name] = df[column_name].fillna(median_values)
    else:
        print("Invalid method specified")


def imputate_missing_values_services():
    filename = '../../datasets/prepared/class_credit_score_encoded_1.csv'
    df = pd.read_csv(filename, na_values="")

    # Drop 400 rows with missing values, then reset indices
    df = mvi_by_dropping(df, min_pct_per_variable=0.0, min_pct_per_record=0.9)
    df.reset_index(drop=True, inplace=True)

    # render_mv(df)
    impute_column(df, 'Age', 'mode')
    impute_column(df, 'Occupation', 'mode')
    impute_column(df, 'Monthly_Inhand_Salary', 'mode')
    impute_column(df, 'NumofDelayedPayment', 'mean')
    impute_column(df, 'ChangedCreditLimit', 'mean', rounding=False)
    impute_column(df, 'NumCreditInquiries', 'mode')

    # render_mv(df)
    df['NumofDelayedPayment'].describe()
    print(df.shape)

    # df.to_csv('../../datasets/prepared/class_credit_score_encoded_2.csv', index=False)

    # KNN and frequent imputation
    '''try:
        df.drop(columns=["Customer_ID"], inplace=True)
    except KeyError:
        pass
    knn = mvi_by_filling(df, 'knn')
    freq = mvi_by_filling(df, 'frequent')
    knn.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_knn.csv', index=False)
    freq.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_freq.csv', index=False)'''


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
    plt.savefig('../../figures/temp/missing_values_per_variable.png')
    show()


if __name__ == "__main__":
    imputate_missing_values_services()
