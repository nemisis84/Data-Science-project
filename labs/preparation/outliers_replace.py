from pandas import read_csv, DataFrame
from helpers.dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
)


def replace_outliers(data: DataFrame):
    numeric_vars: list[str] = get_variable_types(data)["numeric"]
    if [] != numeric_vars:
        df: DataFrame = data.copy(deep=True)
        summary5: DataFrame = data[numeric_vars].describe()
        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var])
            median: float = df[var].median()
            df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
        df.to_csv("../../datasets/prepared/outliers_replaced.csv", index=False)
        print("Data after replacing outliers:", df.shape)
        print(df.describe())
    else:
        print("There are no numeric variables")







