from matplotlib.pyplot import subplots, figure, savefig
from numpy import array, ndarray, std, argsort
from sklearn.ensemble import RandomForestClassifier

from helpers.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files, plot_horizontal_bar_chart,
)
from helpers.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def random_forests_study(
        trnX: ndarray,
        trnY: array,
        tstX: ndarray,
        tstY: array,
        nr_max_trees: int = 2500,
        lag: int = 500,
        metric: str = "accuracy",
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model_: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for feature in max_features:
            y_test_values: list[float] = []
            for n_ in n_estimators:
                clf_ = RandomForestClassifier(
                    n_estimators=n_, max_depth=d, max_features=feature
                )
                clf_.fit(trnX, trnY)
                prdY: array = clf_.predict(tstX)
                evaluation: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_test_values.append(evaluation)
                if evaluation - best_performance > DELTA_IMPROVE:
                    best_performance = evaluation
                    best_params["params"] = (d, feature, n_)
                    best_model_ = clf_

            values[feature] = y_test_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model_, best_params


if __name__ == "__main__":
    target = "CovidPos"
    train_filename = f"../../datasets/prepared/{target}_train.csv"
    test_filename = f"../../datasets/prepared/{target}_test.csv"
    eval_metric = "accuracy"
    image_path = f"../../figures/modeling/random_forests/{target}_rf_{eval_metric}_"

    trnX, tstX, trnY, tstY, labels, var_s = read_train_test_from_files(
        train_filename, test_filename, target
    )

    print(f"Labels={labels}")

    figure()
    best_model, params = random_forests_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )
    savefig(f"{image_path}_study.png")
    # show()

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'{image_path}_{params["name"]}_best_{params["metric"]}_eval.png')
    # show()

    stdevs: list[float] = list(
        std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    )
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(var_s)):
        elems += [var_s[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f + 1}. {elems[f]} ({importances[indices[f]]})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="RF variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"{image_path}_vars_ranking.png")

    d_max: int = params["params"][0]
    feat: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"RF overfitting study for d={d_max} and f={feat}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f"{image_path}_overfitting.png")
