from sklearn.preprocessing import MinMaxScaler
from pandas import (read_csv, DataFrame, Series)


def scale_minmax(df, target, save=False, file_prefix=""):
    """

    :param df: DataFrame
    :param target: "Credit_Score" or "CovidPos"
    :param save: Boolean
    :param file_prefix: String
    :return: DataFrame
    """
    cols: list[str] = df.columns.to_list()
    target_data: Series = df.pop(target)

    transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df)
    df_minmax = DataFrame(transf.transform(df), index=df.index)
    df_minmax[target] = target_data
    cols.remove(target)
    cols.append(target)
    df_minmax.columns = cols
    if save:
        df_minmax.to_csv(f"../../datasets/prepared/{file_prefix}_4_minmax.csv", index=False)
    return df_minmax
