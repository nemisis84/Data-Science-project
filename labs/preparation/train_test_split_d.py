import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from numpy import array, ndarray
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split

from helpers.dslabs_functions import plot_multibar_chart, evaluate_approach


def split_datasets(df, target, id, path='../../datasets/prepared/split/data'):
    labels = list(df[target].unique())
    labels.sort()
    print(f"Labels={labels}")

    positive: int = 1
    negative: int = 0
    values: dict[str, list[int]] = {
        "Original": [
            len(df[df[target] == negative]),
            len(df[df[target] == positive]),
        ]
    }

    y: array = df.pop(target).to_list()
    x: ndarray = df.values

    trnX, tstX, trnY, tstY = train_test_split(x, y, train_size=0.7, stratify=y)

    train: DataFrame = concat(
        [DataFrame(trnX, columns=df.columns), DataFrame(trnY, columns=[target])], axis=1
    )
    train.to_csv(f"{path}_train_{id}.csv", index=False)

    test: DataFrame = concat(
        [DataFrame(tstX, columns=df.columns), DataFrame(tstY, columns=[target])], axis=1
    )
    test.to_csv(f"{path}_test_{id}.csv", index=False)

    '''
    values["Train"] = [
        len(train[train[target] == negative]),
        len(train[train[target] == positive]),
    ]
    values["Test"] = [
        len(test[test[target] == negative]),
        len(test[test[target] == positive]),
    ]
    '''

    # figure(figsize=(6, 4))
    # plot_multibar_chart(labels, values, title="Data distribution per dataset")
    # show()
    return train, test


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


if __name__ == "__main__":
    # Approach one
    fin1 = '../../datasets/prepared/class_credit_score_2_1.csv'
    # Approach two
    fin2 = '../../datasets/prepared/class_credit_score_2_knn.csv'
    # Target, should be the same for both approaches
    fin_tar = 'Credit_Score'
    # Output path, make sure to create a new folder for the split datasets
    fin_out = '../../datasets/prepared/split/credit_score'

    # run the function, the id is used to differentiate the output files. for MVI I used 'custom' and 'knn'
    # If your dataset has negative values, add neg=True
    split_and_test(fin1, fin_tar, fin_out, 'custom', neg=True)
    split_and_test(fin2, fin_tar, fin_out, 'knn', neg=True)

    # Then do the same for the health datasets
    cov1 = '../../datasets/prepared/class_pos_covid_2_1.csv'
    cov2 = '../../datasets/prepared/class_pos_covid_2_knn.csv'

    cov_tar = 'CovidPos'
    cov_out = '../../datasets/prepared/split/covid_pos'

    split_and_test(cov1, cov_tar, cov_out, 'custom', neg=True)
    split_and_test(cov2, cov_tar, cov_out, 'knn', neg=True)

