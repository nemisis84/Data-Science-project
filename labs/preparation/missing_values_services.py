import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure, show
from helpers.dslabs_functions import plot_bar_chart, get_variable_types, define_grid, HEIGHT, plot_multibar_chart, \
    determine_outlier_thresholds_for_var, count_outliers, derive_date_variables, analyse_date_granularity, \
    mvi_by_filling
import seaborn as sns

occupations = ['Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
               'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
               'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
               'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer']


# This misses 70 entries
# We assume that people are not unemployed, so we check if they have at least one occupation.
def check_occupation_row(row, df):
    for occupation in occupations:
        if row[occupation] == 1:
            return

    # Check row above
    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
        for occupation in occupations:
            if df[occupation].iloc[row.name - 1] == 1:
                df[occupation].iloc[row.name] = 1
                return

    # Check row below
    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
        for occupation in occupations:
            if df[occupation].iloc[row.name + 1] == 1:
                df[occupation].iloc[row.name] = 1
                return


# This catches all entries
# Might push birthdays a couple of months forward.
def check_age_row(row, df):
    age = row['Age']

    if pd.isna(age) or age > 122:
        # Check row below
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
            age = df['Age'].iloc[row.name + 1]
            if pd.notna(age):
                if age < 122:
                    df['Age'].iloc[row.name] = age
                    if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
                        df['Age'].iloc[row.name - 1] = age
                    return

        # Check row above
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
            age = df['Age'].iloc[row.name - 1]
            if pd.notna(age):
                if age < 122:
                    df['Age'].iloc[row.name] = age
                    return


# This misses 58 entries
# We assume that people are not unemployed. !!!
def check_monthly_inhand_salary_row(row, df):
    monthly_inhand_salary = row['Monthly_Inhand_Salary']

    if pd.isna(monthly_inhand_salary):
        # Check row below
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name + 1]:
            monthly_inhand_salary = df['Monthly_Inhand_Salary'].iloc[row.name + 1]
            if pd.notna(monthly_inhand_salary):
                df['Monthly_Inhand_Salary'].iloc[row.name] = monthly_inhand_salary
                if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
                    df['Monthly_Inhand_Salary'].iloc[row.name - 1] = monthly_inhand_salary
                return

        # Check row above
        if row['Customer_ID'] == df['Customer_ID'].iloc[row.name - 1]:
            monthly_inhand_salary = df['Monthly_Inhand_Salary'].iloc[row.name - 1]
            if pd.notna(monthly_inhand_salary):
                df['Monthly_Inhand_Salary'].iloc[row.name] = monthly_inhand_salary
                return


def imputate_missing_values_services():
    filename = '../../datasets/prepared/class_credit_score_encoded_1.csv'
    df = pd.read_csv(filename, na_values="")

    try:
        df.drop(columns=["Customer_ID"], inplace=True)
    except KeyError:
        pass

    # knn = mvi_by_filling(df, 'knn')
    # const = mvi_by_filling(df, 'constant')
    # freq = mvi_by_filling(df, 'frequent')
    # knn.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_knn.csv', index=False)
    # const.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_const.csv', index=False)
    # freq.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_freq.csv', index=False)


def render_mv(df):
    print(f"Dataset nr records={df.shape[0]}", f"nr variables={df.shape[1]}")

    mv: dict[str, int] = {}
    for var in df.columns:
        nr: int = df[var].isna().sum()
        if nr > 0:
            mv[var] = nr
        elif var == "Customer_ID":
            mv[var] = 0

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
    show()


if __name__ == "__main__":
    imputate_missing_values_services()
