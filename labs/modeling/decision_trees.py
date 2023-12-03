from typing import Literal
from numpy import array, ndarray, argsort
from matplotlib.pyplot import figure, savefig, show, close
from sklearn.tree import DecisionTreeClassifier, plot_tree
from helpers.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files, \
    plot_horizontal_bar_chart
from helpers.dslabs_functions import plot_evaluation_results, plot_multiline_chart


def trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int = 10, lag: int = 2, metric='accuracy'
) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
    plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

    return best_model, best_params


def do_decision_trees(best_model, params, target, metric, trnX, tstX, trnY, tstY, labels):
    figure()
    savefig(f'../../figures/modeling/{target}_dt_{metric}_study.png')
    # show()
    # close()

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    # show()
    savefig(f'../../figures/modeling/{target}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
    # close()


def overfitting_study(params, target, metric, trnX, tstX, trnY, tstY):
    figure()

    crit: Literal["entropy", "gini"] = params["params"][0]
    d_max = 25
    depths: list[int] = [i for i in range(2, d_max + 1, 1)]
    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric = "accuracy"
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study for {crit}",
        xlabel="max_depth",
        ylabel=str(metric),
        percentage=True,
    )
    savefig(f"../../figures/modeling/{target}_dt_{metric}_overfitting.png")


def var_importance(best_model, vars, target, metric):
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f"{f + 1}. {elems[f]} ({importances[indices[f]]})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Decision Tree variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"../../figures/modeling/{target}_dt_{metric}_vars_ranking.png")


def plot_dec_trees(best_model, vars, labels, target, metric):
    tree_filename: str = f"../../figures/modeling/{target}_dt_{metric}_best_tree"
    max_depth2show = 3
    st_labels: list[str] = [str(value) for value in labels]
    figure(figsize=(14, 6))
    plot_tree(
        best_model,
        max_depth=max_depth2show,
        feature_names=vars,
        class_names=st_labels,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
    )
    savefig(tree_filename + ".png")


if __name__ == "__main__":
    train_f = "../../datasets/prepared/Credit_Score_train.csv"
    test_f = "../../datasets/prepared/Credit_Score_test.csv"
    target_s = "Credit_Score"
    # metric_s = "accuracy"
    for metric_s in ['accuracy', 'recall', 'precision']:
        trnX_s, tstX_s, trnY_s, tstY_s, labels_s, vars_s = read_train_test_from_files(train_f, test_f, target_s)
        best_model_s, params_s = trees_study(trnX_s, trnY_s, tstX_s, tstY_s, d_max=25, metric=metric_s)

        do_decision_trees(best_model_s, params_s, target_s, metric_s, trnX_s, tstX_s, trnY_s, tstY_s, labels_s)
        overfitting_study(params_s, target_s, metric_s, trnX_s, tstX_s, trnY_s, tstY_s)
        var_importance(best_model_s, vars_s, target_s, metric_s)
        plot_dec_trees(best_model_s, vars_s, labels_s, target_s, metric_s)
