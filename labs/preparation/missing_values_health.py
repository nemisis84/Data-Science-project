from helpers.dslabs_functions import mvi_by_dropping
from labs.preparation.missing_values_functions import impute_column


def init_impute(data):
    # Drops no variables, but drops 0.3% of the records (1 286)
    df = mvi_by_dropping(data, min_pct_per_variable=0.9, min_pct_per_record=0.85)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)

    return df


def impute_health(data, save=False):
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
    impute_column(df, 'DifficultyDressingBathing', 'mode')
    impute_column(df, 'DifficultyWalking', 'mode')
    impute_column(df, 'DifficultyErrands', 'mode')
    impute_column(df, 'SmokerStatus', 'mode')
    impute_column(df, 'ECigaretteUsage', 'mode')
    impute_column(df, 'ChestScan', 'mode')
    impute_column(df, 'AgeCategory', 'mode')
    impute_column(df, 'RaceEthnicityCategory', 'mode')
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
        df.to_csv('../../datasets/prepared/2_CovidPos.csv', index=False)

    return df
