from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler

# TODO: replace with csv after outlier treatment
data: DataFrame = read_csv("../../datasets/prepared/class_credit_score_encoded_1.csv")
target = "Credit_Score"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv("../../datasets/prepared/class_credit_score_4_zscore.csv", index=False)