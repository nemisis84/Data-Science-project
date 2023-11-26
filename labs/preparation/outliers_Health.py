import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import savefig, show, figure
from helpers.dslabs_functions import (IQR_FACTOR, NR_STDEV, count_outliers, get_variable_types, plot_multibar_chart, 
                                    determine_outlier_thresholds_for_var)

def drop_outliers(df, cols, std: bool = True, thres: int=NR_STDEV):
    if cols is not None:
        print(f"Original data: {df.shape}")
        data: DataFrame = df.copy(deep=True)
        summary5: DataFrame = data[cols].describe()
        for var in cols:
            top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
                summary5[var], std_based = std, threshold = thres
            )
            outliers: Series = data[(data[var] > top_threshold) | (data[var] < bottom_threshold)]
            n = len(data.loc[(data[var] > top_threshold) | (data[var] < bottom_threshold)])
            data.drop(outliers.index, axis=0, inplace=True)
        print(f"Data after dropping outliers: {data.shape}")
        return data
    
    else:
        return(print("There are no numeric variables"))
    

def replace_outliers(df, cols, std: bool = True, thres: int=NR_STDEV):
    if cols is not None:
        #print(f"Original data: {df.shape}")
        data: DataFrame = df.copy(deep=True)
        summary5: DataFrame = data[cols].describe()
        for var in cols:
            top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
                summary5[var], std_based = std, threshold = thres
            )
            #n = len(data.loc[(data[var] > top_threshold) | (data[var] < bottom_threshold)])
            #print(f"Number of replacing values for {var}: {n}")
            median: float = data[var].median()
            data[var] = data[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)
        return data
    
    else:
        return(print("There are no numeric variables"))

    
health_vars_drop_outliers = ['HeightInMeters', 'WeightInKilograms', 'BMI']
health_vars_replace_outliers = ['SleepHours']

services_vars_drop_outliers = ['Annual_Income', 'MonthlyBalance', 'NumofDelayedPayment']
services_vars_replace_outliers = ['Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate',
                                  'NumofLoan','Delay_from_due_date','NumofDelayedPayment','ChangedCreditLimit','NumCreditInquiries',
                                  'OutstandingDebt','CreditUtilizationRatio','TotalEMIpermonth','Amountinvestedmonthly','MonthlyBalance']

def outliers_health():
    df = pd.read_csv('../../datasets/class_pos_covid.csv')

    df_health1 = drop_outliers(df, health_vars_drop_outliers, std = False, thres = IQR_FACTOR)
    df_health1 = replace_outliers(df_health1, health_vars_replace_outliers, std = False, thres = IQR_FACTOR)

    df_health2 = drop_outliers(df, health_vars_drop_outliers)
    df_health2 = replace_outliers(df_health2, health_vars_replace_outliers)

    df_health1.to_csv("../../datasets/prepared/class_pos_covid_outliers_iqr.csv")
    df_health1.to_csv("../../datasets/prepared/class_pos_covid_outliers_std.csv")


def outliers_services():
    df = pd.read_csv('../../datasets/class_credit_score.csv')

    df_services1 = drop_outliers(df, services_vars_drop_outliers, std = False, thres = IQR_FACTOR)
    df_services1 = replace_outliers(df_services1, services_vars_replace_outliers, std = False, thres = IQR_FACTOR)

    df_services2 = drop_outliers(df, services_vars_drop_outliers)
    df_services2 = replace_outliers(df_services2, services_vars_replace_outliers)

    df_services1.to_csv("../../datasets/prepared/class_credit_score_outliers_iqr.csv")
    df_services1.to_csv("../../datasets/prepared/class_credit_score_outliers_std.csv")

if __name__ == "__main__":
    outliers_health()
    outliers_services()