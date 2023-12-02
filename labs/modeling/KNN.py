from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show, clf
from helpers.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from helpers.dslabs_functions import read_train_test_from_files, plot_evaluation_results

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy'
        ) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

    kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params['params'] = (k, d)
                best_model = clf
            #print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params

def KNN(train_set, test_set, target, k_max: int=19, metric: str = "accuracy"):
    trnX: ndarray
    tstX: ndarray
    trnY: array
    tstY: array
    labels: list
    vars: list
    
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_set, test_set, target
    )

    print(f'Train#={len(trnX)} Test#={len(tstX)}')
    print(f'Labels={labels}')

    figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max = k_max, metric = metric)
    savefig(f"../../figures/{target}/Evaluation/KNN_{metric}_study.png")
    plt.clf()

    prd_trn: array = best_model.predict(trnX)    
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'../../figures/{target}/Evaluation/KNN_best_model_{params['params'][0]}_{params['params'][1]}.png')
    plt.clf()

    distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
    K_MAX = 25
    kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
    y_tst_values: list = []
    y_trn_values: list = []
    acc_metric: str = metric
    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        kvalues,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"KNN overfitting study for {distance}",
        xlabel="K",
        ylabel=str(metric),
        percentage=True,
        )
    savefig(f'../../figures/{target}/Evaluation/KNN_overfitting.png')



if __name__ == "__main__":
    CovidPos_train = '../../datasets/tests/7_CovidPos_train.csv'
    CovidPos_test = '../../datasets/tests/6_CovidPos_select_features__test_variance.csv'
    CovidPos_target = 'CovidPos'

    KNN(CovidPos_train, CovidPos_test, CovidPos_target, k_max = 25)




