from numpy import ndarray
from imblearn.over_sampling import SMOTE
import pandas as pd

def sampling(df, target, path=False, sampling="oversampling"):
    print(f"\nBalancing using sampling strategy '{sampling}':\n")

    target_count = df[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    
    print("Minority class=", positive_class, ":", target_count[positive_class])
    print("Majority class=", negative_class, ":", target_count[negative_class])
    print(
        "Proportion:",
        round(target_count[positive_class] / target_count[negative_class], 2),
        ": 1",
    )
    print()

    df_positives = df[df[target] == positive_class]
    df_negatives = df[df[target] == negative_class]

    if sampling == "oversampling":
        df_pos_sample = pd.DataFrame(df_positives.sample(len(df_negatives), replace=True))
        df_sampled = pd.concat([df_pos_sample, df_negatives], axis=0)

        print("Minority class=", positive_class, ":", len(df_pos_sample))
        print("Majority class=", negative_class, ":", len(df_negatives))
        print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")

    elif sampling == "undersampling":
        df_neg_sample = pd.DataFrame(df_negatives.sample(len(df_positives)))
        df_sampled = pd.concat([df_positives, df_neg_sample], axis=0)
        
        print("Minority class=", positive_class, ":", len(df_positives))
        print("Majority class=", negative_class, ":", len(df_neg_sample))
        print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")

    else:
        print("No valid sampling method")
        return
    
    if path:
        df_sampled.to_csv(path, index=False)
    print(f"Shape of table: {df_sampled.shape}")
    return df_sampled

def SMOTE_balancing(df, target, RANDOM_STATE, path = False, sampling_strategy = "minority"):
    print(f"\nBalancing using SMOTE with sampling strategy '{sampling_strategy}':\n")
    
    target_count = df[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    print("Inital proportion:")
    print("Minority class=", positive_class, ":", target_count[positive_class])
    print("Majority class=", negative_class, ":", target_count[negative_class])
    print(
        "Proportion:",
        round(target_count[positive_class] / target_count[negative_class], 2),
        ": 1",
    )
    print()
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    y = df.pop(target).values
    X: ndarray = df.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = pd.concat([pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df.columns) + [target]
    if path:
        df_smote.to_csv(path, index=False)

    smote_target_count = pd.Series(smote_y).value_counts()

    print("After balancing")
    print("Minority class=", positive_class, ":", smote_target_count[positive_class])
    print("Majority class=", negative_class, ":", smote_target_count[negative_class])
    print(
        "Proportion:",
        round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
        ": 1",
    )
    print(f"Shape of table: {df_smote.shape}")
    return df_smote


if __name__ == "__main__":
    df = pd.read_csv('../../datasets/prepared/class_credit_score_encoded_1.csv')
    df = df.dropna()

    sampling(df, "Credit_Score", sampling="undersampling")
    SMOTE_balancing(df, "Credit_Score", 85)