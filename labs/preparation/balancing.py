from numpy import ndarray
from imblearn.over_sampling import SMOTE
import pandas as pd


def sampling(df, target, path=False, sampling="oversampling"):
    target_count = df[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = df[df[target] == positive_class]
    df_negatives = df[df[target] == negative_class]

    if sampling == "oversampling":
        df_pos_sample = pd.DataFrame(df_positives.sample(len(df_negatives), replace=True))
        df_sampled = pd.concat([df_pos_sample, df_negatives], axis=0)

    elif sampling == "undersampling":
        df_neg_sample = pd.DataFrame(df_negatives.sample(len(df_positives)))
        df_sampled = pd.concat([df_positives, df_neg_sample], axis=0)
    else:
        print("No valid sampling method")
        return

    if path:
        df_sampled.to_csv(path, index=False)
    return df_sampled


def SMOTE_balancing(df, target, RANDOM_STATE, path=False, sampling_strategy="minority"):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    y = df.pop(target).values
    X: ndarray = df.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = pd.concat([pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df.columns) + [target]
    if path:
        df_smote.to_csv(path, index=False)

    return df_smote


if __name__ == "__main__":
    df = pd.read_csv('../../datasets/prepared/split/class_credit_score_test_credit_score.csv')
    tar = 'Credit_Score'
    sampling(df.copy(deep=True), tar, sampling="undersampling",
             path="../../datasets/prepared/class_credit_score_6_undersampled")
    sampling(df.copy(deep=True), tar, path="../../datasets/prepared/class_credit_score_6_oversampled")

    SMOTE_balancing(df.copy(), tar, 85, '../../datasets/prepared/class_credit_score_6_SMOTE')
