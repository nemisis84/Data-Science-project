from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from helpers.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from helpers.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart
from numpy import std, argsort
from helpers.dslabs_functions import plot_horizontal_bar_chart
import matplotlib.pyplot as plt

def gradient_boosting_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[GradientBoostingClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [5, 8, 11]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = GradientBoostingClassifier(
                    n_estimators=n, max_depth=d, learning_rate=lr
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, lr, n)
                    best_model = clf
                # print(f'GB d={d} lr={lr} n={n}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

def gradient_boosting_variables_studies(best_model, vars, path):
    trees_importances: list[float] = []
    for lst_trees in best_model.estimators_:
        for tree in lst_trees:
            trees_importances.append(tree.feature_importances_)

    stdevs: list[float] = list(std(trees_importances, axis=0))
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    plt.figure(figsize=(9, 8))
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="GB variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    if path:
        plt.yticks(rotation=45)
        savefig(path+"gradient_boosting_variables_study")

def gradient_boosting_overfitting_study(trnX, trnY, eval_metric = "accuracy", path=False):
    d_max: int = params["params"][0]
    lr: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"GB overfitting study for d={d_max} and lr={lr}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    if path:
        savefig(path+"gradient_boosting_overfitting_study")

def study_and_train(train_filename, test_filename, target, path = False, eval_metric = "accuracy"):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_filename, test_filename, target
    )
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")

    figure()
    best_model, params = gradient_boosting_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )
    if path:
        savefig(path+"gradient_boosting_study")
    print("Training ...")
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    if path:
        plt.tight_layout()
        savefig(path+"gradient_boosting_best_results")
    
    return best_model, params, trnX, tstX, trnY, tstY, vars

if __name__ == "__main__":
    
    dataset_path = "../../datasets/prepared/"

    # Covid dataset
    train_filename = dataset_path + "7_CovidPos_train.csv"
    test_filename = dataset_path + "6_CovidPos_select_features__test_variance.csv"
    target = "CovidPos"
    figure_path = "../../figures/modeling/gradient_boosting/CovidPos_"
    best_model, params, trnX, tstX, trnY, tstY, vars = study_and_train(train_filename, test_filename, target, path = figure_path)
    gradient_boosting_variables_studies(best_model, vars, path=figure_path)
    gradient_boosting_overfitting_study(trnX, trnY, path=figure_path)


    # Credit dataset
    train_filename = dataset_path + "7_Credit_Score_train.csv"
    test_filename = dataset_path +"6_Credit_Score_select_features__test_variance.csv"
    target = "Credit_Score"
    figure_path = "../../figures/modeling/gradient_boosting/Credit_Score_"
    best_model, params, trnX, tstX, trnY, tstY, vars = study_and_train(train_filename, test_filename, target, path = figure_path)
    gradient_boosting_variables_studies(best_model, vars, path=figure_path)
    gradient_boosting_overfitting_study(trnX, trnY, path=figure_path)


