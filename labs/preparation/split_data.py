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
    plt.figure()
    eval = evaluate_approach(train, test, target, neg=neg)
    print(eval)
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{target} {id} evaluation", percentage=True
    )
    plt.savefig(f"{output}_evaluation_{id}.png")

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
