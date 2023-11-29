import pandas as pd
from sklearn.model_selection import train_test_split

from helpers.dslabs_functions import (
    select_low_variance_variables,
    apply_feature_selection,
    select_redundant_variables
)
from split_data import only_eval


def select_variables(df, target, path, method='redundant', param=0.5):
    train, test = train_test_split(df, train_size=0.7)

    if method == 'variance':
        print("Original variables", train.columns.to_list())
        vars2drop: list[str] = select_low_variance_variables(train, param, target=target)
        print("\n\nVariables to drop", vars2drop)

    elif method == 'redundant':
        print("Original variables", train.columns.values)
        vars2drop: list[str] = select_redundant_variables(train, min_threshold=param, target=target)
        print("\n\nVariables to drop", vars2drop)

    train_cp, test_cp = apply_feature_selection(
        train, test, vars2drop, filename=f"{path}", tag=method
    )
    print(f"\n\nOriginal data: train={train.shape}, test={test.shape}")
    print(f"After {method} FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")
    return train_cp, test_cp


if __name__ == "__main__":
    # Credit_score
    df = pd.read_csv('../../datasets/prepared/class_credit_score_4_zscore.csv')
    target = "Credit_Score"
    path = '../../datasets/feature_selection/Credit_Score_featureSelection'
    # Variance
    train_cp, test_cp = select_variables(df, target, path, method='variance', param=1)
    only_eval(train_cp, test_cp, target, path, 'Variance = 1', neg=True)

    train_cp, test_cp = select_variables(df, target, path, method='variance', param=0.5)
    only_eval(train_cp, test_cp, target, path, 'Variance = 0.5', neg=True)
    # Reduncancy
    train_cp, test_cp = select_variables(df, target, path, method='redundant', param=0.25)
    only_eval(train_cp, test_cp, target, path, 'Redundancy = 0.25', neg=True)

    train_cp, test_cp = select_variables(df, target, path, method='redundant', param=0.5)
    only_eval(train_cp, test_cp, target, path, 'Redundancy = 0.5', neg=True)

    # Health
    df = pd.read_csv('../../datasets/prepared/class_covidpos_4_zscore.csv')
    target = "CovidPos"
    path = '../../datasets/feature_selection/CovidPos_featureSelection'
    # Variance
    train_cp, test_cp = select_variables(df, target, path, method='variance', param=1)
    only_eval(train_cp, test_cp, target, path, 'Variance = 1', neg=True)

    train_cp, test_cp = select_variables(df, target, path, method='variance', param=0.5)
    only_eval(train_cp, test_cp, target, path, 'Variance = 0.5', neg=True)
    # Reduncancy
    train_cp, test_cp = select_variables(df, target, path, method='redundant', param=0.25)
    only_eval(train_cp, test_cp, target, path, 'Redundancy = 0.25', neg=True)

    train_cp, test_cp = select_variables(df, target, path, method='redundant', param=0.4)
    only_eval(train_cp, test_cp, target, path, 'Redundancy = 0.5', neg=True)
