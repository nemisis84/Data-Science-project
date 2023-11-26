from pandas import read_csv, DataFrame
from helpers.dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
)


def truncate_outliers(data: DataFrame):
    numeric_vars: list[str] = get_variable_types(data)["numeric"]
    if [] != numeric_vars:
        df: DataFrame = data.copy(deep=True)
        summary5: DataFrame = data[numeric_vars].describe()
        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var])
            df[var] = df[var].apply(
                lambda x: top if x > top else bottom if x < bottom else x
            )
        df.to_csv("../../datasets/prepared/outliers_truncated.csv", index=True)
        print("Data after truncating outliers:", df.shape)
        print(df.describe())
    else:
        print("There are no numeric variables")

truncate_outliers(read_csv("../../datasets/prepared/class_pos_covid_encoded_1.csv"))
