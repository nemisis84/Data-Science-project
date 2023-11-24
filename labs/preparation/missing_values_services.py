import pandas as pd

from helpers.dslabs_functions import mvi_by_filling


def imputate_missing_values_services():
    filename = '../../datasets/prepared/class_credit_score_encoded_1.csv'
    df = pd.read_csv(filename, na_values="")

    try:
        df.drop(columns=["Customer_ID"], inplace=True)
    except KeyError:
        pass

    knn = mvi_by_filling(df, 'knn')
    knn.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_knn.csv', index=False)

    freq = mvi_by_filling(df, 'frequent')
    freq.to_csv('../../datasets/prepared/class_credit_score_encoded_2_MVI_freq.csv', index=False)


if __name__ == "__main__":
    imputate_missing_values_services()
