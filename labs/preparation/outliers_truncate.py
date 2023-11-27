from pandas import DataFrame
from helpers.dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
)


def truncate_outliers(data: DataFrame, save=False, file_prefix=""):
    numeric_vars: list[str] = get_variable_types(data)["numeric"]

    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[numeric_vars].describe()
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        df[var] = df[var].apply(
            lambda x: top if x > top else bottom if x < bottom else x
        )
    if save:
        df.to_csv(f"../../datasets/prepared/{file_prefix}_3_truncated.csv", index=False)
    print("Data after truncating outliers:", df.shape)
    print(df.describe())
    return df
