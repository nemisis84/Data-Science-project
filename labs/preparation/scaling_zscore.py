from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler


def scale_zscore(data: DataFrame, target):
    """

    :param data: DataFrame
    :param target: "Credit_Score" or "CovidPos"
    :param save: Boolean
    :param file_prefix: String
    :return: DataFrame
    """
    cols: list[str] = data.columns.to_list()
    target_data: Series = data.pop(target)

    transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
    df_zscore = DataFrame(transf.transform(data), index=data.index)
    df_zscore[target] = target_data
    cols.remove(target)
    cols.append(target)
    df_zscore.columns = cols

    df_zscore.to_csv(f"../../datasets/prepared/4_{target}.csv", index=False)

    return df_zscore
