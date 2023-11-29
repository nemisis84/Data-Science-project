import pandas as pd
from matplotlib import pyplot as plt

from data_encoding_health import encode_health
from data_encoding_Services import encode_services
from missing_values_health import impute_health
from missing_values_services import impute_services
from outliers_truncate import truncate_outliers
from scaling_zscore import scale_zscore
from split_data import split_datasets
from balancing import SMOTE_balancing, sampling
from feature_selection import select_variables
from helpers.dslabs_functions import evaluate_approach, plot_multibar_chart


def encode_data(df_in, target):
    if 'Covid' in target:
        df = encode_health(df_in)
    else:
        df = encode_services(df_in)

    return df


def imputate_data(df_in, target):
    if 'CovidPos' in target:
        df = impute_health(df_in, save=True)
    else:
        df = impute_services(df_in, save=True)

    return df


def outlier_handling(df, target):
    truncate_outliers(df, file_prefix=target)

    return df


def scale_data(df_in, target):
    df = scale_zscore(df_in, target)

    return df


def select_features(df_in, target):
    path = f"../../datasets/prepared/6_{target}_select_features_"
    train, test = select_variables(df_in, target, path, method="variance", param=0.5)
    return train, test


def split_dataset(df_in, target):
    train, test = split_datasets(df_in, target)
    return [train, test]


def balance_training(train_in, target):
    path = f"../../datasets/prepared/7_{target}_train.csv"
    #if 'CovidPos' in target:
    train_out = sampling(train_in, target, path=path, sampling="undersampling")
    #else:
    #    train_out = SMOTE_balancing(train_in, target, 42, path=path, sampling_strategy="minority")

    return train_out


def eval_dataset(train, test, target):
    eval = evaluate_approach(train, test, target, neg=True)
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{target} evaluation", percentage=True
    )
    plt.savefig(f"../../figures/preparation/8_{target}_evaluation.png")
    plt.show()


def main(file):
    if 'Credit_Score' in file:
        target = 'Credit_Score'
    elif 'CovidPos' in file:
        target = 'CovidPos'
    else:
        print("No target found.\nExiting")
        exit(1)

    print(f"----------------------------------------\n"
          f"Starting pipeline for {target}...")
    df = pd.read_csv(file)
    print("Encoding...")
    df = encode_data(df, target)
    print("Imputating...")
    df = imputate_data(df, target)
    print("Outlier handling...")
    df = outlier_handling(df, target)
    print("Scaling...")
    df = scale_data(df, target)
    print("Feature selection...")
    train, test = select_features(df, target)
    print("Balancing...")
    train = balance_training(train, target)
    print("Evaluating...")
    eval_dataset(train, test, target)

    print(f"Pipeline for {target} finished.\n\n")


if __name__ == "__main__":
    main('../../datasets/Credit_Score.csv')
    main('../../datasets/CovidPos.csv')
