from helpers.dslabs_functions import (
    select_low_variance_variables,
    apply_feature_selection,
    select_redundant_variables
)
from sklearn.model_selection import train_test_split
import pandas as pd

def select_variables(df, target, path, method='redundant'):
    train, test = train_test_split(df, train_size=0.7)
    
    if method == 'variance':
        print("Original variables", train.columns.to_list())
        vars2drop: list[str] = select_low_variance_variables(train, 3, target=target)
        print("Variables to drop", vars2drop)  

    elif method == 'redundant':
        print("Original variables", train.columns.values)
        vars2drop: list[str] = select_redundant_variables(train, min_threshold=0.5, target=target)
        print("Variables to drop", vars2drop)

    train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename=f"{path}", tag="redundant"
        )
    print(f"Original data: train={train.shape}, test={test.shape}")
    print(f"After {method} FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")
    return train_cp, test_cp


if __name__ == "__main__":
    df = pd.read_csv('../../datasets/prepared/class_pos_covid_outliers_iqr.csv')
    df = df.dropna()
    path = '../../datasets/prepared/'
    target = 'CovidPos'
    select_variables(df, target, path, method='variance')

