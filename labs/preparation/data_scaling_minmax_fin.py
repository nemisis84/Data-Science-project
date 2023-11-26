from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame, Series

# TODO: replace with csv after outlier treatment
data: DataFrame = read_csv("../../datasets/prepared/class_credit_score_encoded_1.csv")
target = "Credit_Score"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv("../../datasets/prepared/class_credit_score_4_minmax.csv", index=False)
