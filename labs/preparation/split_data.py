import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from numpy import array, ndarray
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split

from helpers.dslabs_functions import plot_multibar_chart, evaluate_approach


def split_datasets(df, target):
    y: array = df.pop(target).to_list()
    x: ndarray = df.values

    trnX, tstX, trnY, tstY = train_test_split(x, y, train_size=0.7, stratify=y, random_state=42)  # Set the seed

    train: DataFrame = concat(
        [DataFrame(trnX, columns=df.columns), DataFrame(trnY, columns=[target])], axis=1
    )
    train.to_csv(f"../../datasets/prepared/6_{target}_train.csv", index=False)

    test: DataFrame = concat(
        [DataFrame(tstX, columns=df.columns), DataFrame(tstY, columns=[target])], axis=1
    )
    test.to_csv(f"../../datasets/prepared/6_{target}_test_csv", index=False)

    return train, test

def only_eval(train, test, target, output, id, neg=False):
    eval = evaluate_approach(train, test, target, neg=neg)
    print(eval)
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{target} {id} evaluation", percentage=True
    )
    plt.savefig(f"{output}_evaluation_{id}.png")
    show()

def split_and_test(filename, target, output, id, neg=False):
    df = pd.read_csv(filename, na_values="")

    train, test = split_datasets(df, target, id, output)

    eval = evaluate_approach(train, test, target, neg=neg)
    print(eval)
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{target} {id} evaluation", percentage=True
    )
    plt.savefig(f"{output}_evaluation_{id}.png")
    show()

### Needs to be updated to fit the datasets you want to test
if __name__ == "__main__":
    splitThis = '../../datasets/prepared/class_credit_score_4_zscore.csv'

    # Approach one
    fin1 = '../../datasets/prepared/class_credit_score_6_undersampled'  # Marginally better
    # Approach two
    fin2 = '../../datasets/prepared/class_credit_score_6_oversampled'
    fin3 = '../../datasets/prepared/class_credit_score_6_SMOTE'
    # Target, should be the same for both approaches
    fin_tar = 'Credit_Score'
    # Output path, make sure to create a new folder for the split datasets
    fin_out = '../../datasets/prepared/split/class_credit_score'

    split_datasets(pd.read_csv(splitThis, na_values=""), fin_tar, 'credit_score', fin_out)

    test_file = '../../datasets/prepared/split/class_credit_score_test_credit_score.csv'
    test1 = pd.read_csv(test_file, na_values="")
    #only_eval(pd.read_csv(fin1, na_values=""), test1, fin_tar, fin_out, 'undersampling', neg=True)
    only_eval(pd.read_csv(fin2, na_values=""), test1, fin_tar, fin_out, 'oversampling', neg=True)
    # only_eval(pd.read_csv(fin3, na_values=""), test1, fin_tar, fin_out, 'SMOTE', neg=True)


    # run the function, the id is used to differentiate the output files. for MVI I used 'custom' and 'knn'
    # If your dataset has negative values, add neg=True
    #split_and_test(fin1, fin_tar, fin_out, 'undersampling', neg=True)
    #split_and_test(fin2, fin_tar, fin_out, 'oversampling', neg=True)
    #split_and_test(fin3, fin_tar, fin_out, 'SMOTE', neg=True)
    # Then do the same for the health datasets
    #cov1 = '../../datasets/prepared/class_pos_covid_2_1.csv'
    #cov2 = '../../datasets/prepared/class_pos_covid_2_knn.csv'

    #cov_tar = 'CovidPos'
    #cov_out = '../../datasets/prepared/split/covid_pos'

    #split_and_test(cov1, cov_tar, cov_out, 'custom', neg=True)
    #split_and_test(cov2, cov_tar, cov_out, 'knn', neg=True)

