import pandas as pd

from helpers.dslabs_functions import mvi_by_dropping, mvi_by_filling
from labs.preparation.missing_values_functions import impute_credithistory, impute_column


def init_impute(data):
    # Drops no variables, but drops 0.2% of the records (1117)
    df = mvi_by_dropping(data, min_pct_per_variable=0.9, min_pct_per_record=0.85)
    df.reset_index(drop=True, inplace=True)

    return df

def imputate_health_custom(data, save=False):
    df = init_impute(data)

    impute_column(df, 'GeneralHealth', 'mode')
    impute_column(df, 'PhysicalHealthDays', 'median', rounding=0)  # mean falls outside the 75th percentile
    impute_column(df, 'MentalHealthDays', 'mean', rounding=0)
    impute_column(df, 'LastCheckupTime', 'mode')
    impute_column(df, 'PhysicalActivities', 'mode')
    impute_column(df, 'SleepHours', 'mean', rounding=0)
    impute_column(df, 'RemovedTeeth', 'mode')
    impute_column(df, 'HadHeartAttack', 'mode')
    impute_column(df, 'HadAngina', 'mode')
    impute_column(df, 'HadStroke', 'mode')
    impute_column(df, 'HadAsthma', 'mode')
    impute_column(df, 'HadSkinCancer', 'mode')
    impute_column(df, 'HadCOPD', 'mode')
    impute_column(df, 'HadDepressiveDisorder', 'mode')
    impute_column(df, 'HadKidneyDisease', 'mode')
    impute_column(df, 'HadArthritis', 'mode')
    impute_column(df, 'HadDiabetes', 'mode')
    impute_column(df, 'DeafOrHardOfHearing', 'mode')
    impute_column(df, 'BlindOrVisionDifficulty', 'mode')
    impute_column(df, 'DifficultyConcentrating', 'mode')
    impute_column(df, 'DifficultyWalking', 'mode')
    impute_column(df, 'DifficultyErrands', 'mode')
    impute_column(df, 'SmokerStatus', 'mode')
    impute_column(df, 'ECigaretteUsage', 'mode')
    impute_column(df, 'ChestScan', 'mode')
    impute_column(df, 'AgeCategory', 'mode')
    impute_column(df, 'HeightInMeters', 'mean', rounding=2)
    impute_column(df, 'WeightInKilograms', 'mean', rounding=2)
    impute_column(df, 'BMI', 'median', rounding=2)
    impute_column(df, 'AlcoholDrinkers', 'mode')
    impute_column(df, 'HIVTesting', 'mode')
    impute_column(df, 'FluVaxLast12', 'mode')
    impute_column(df, 'PneumoVaxEver', 'mode')
    impute_column(df, 'TetanusLast10Tdap', 'mode')
    impute_column(df, 'HighRiskLastYear', 'mode')

    if save:
        df.to_csv('../../datasets/prepared/class_pos_covid_2_1.csv', index=False)

    return df


def impute_health_knn(data, save=False):
    df = init_impute(data)
    df = mvi_by_filling(df, 'knn')
    if save:
        df.to_csv('../../datasets/prepared/class_pos_covid_2_knn.csv', index=False)
    return df


if __name__ == "__main__":
    cov = pd.read_csv('../../datasets/prepared/class_pos_covid_encoded_1.csv', na_values="")
    # imputate_health_custom(fin, save=True)
    # impute_health_knn(fin,save=True)

