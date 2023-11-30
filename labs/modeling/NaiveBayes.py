from helpers.dslabs_functions import read_train_test_from_files, plot_confusion_matrix, plot_evaluation_results, CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart
from numpy import array, ndarray
from matplotlib.pyplot import figure, show, savefig,clf
from typing import Callable
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def naive_Bayes_study(
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        #"MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


def NaiveBayes(train_set, test_set, target, metric: str = "accuracy"):
    trnX: ndarray
    tstX: ndarray
    trnY: array
    tstY: array
    labels: list
    vars: list
    
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_set, test_set, target
    )

    figure(figsize=(3.5, 4))
    best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, metric)
    savefig(f"../../figures/modelling/NB_{target}_{metric}_study.png")
    clf()


def best_model_results(train_set, test_set, target, approach, model_name, metric, params = ()):
    model_description: dict = {"name": approach, "metric": metric, "params": params}

    trnX: ndarray
    tstX: ndarray
    trnY: array
    tstY: array
    labels: list
    vars: list
    
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_set, test_set, target
    )

    print(f"Train#={len(trnX)} Test#={len(tstX)}")

    model = model_name
    model.fit(trnX, trnY)

    prd_trn: array = model.predict(trnX)
    prd_tst: array = model.predict(tstX)
    figure()
    plot_evaluation_results(model_description, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'../../figures/modelling/NB_{target}_{model_description["name"]}_best_{model_description["metric"]}_eval.png')
    clf()


if __name__ == "__main__":
    CovidPos_train = '../../datasets/tests/7_CovidPos_train.csv'
    CovidPos_test = '../../datasets/tests/6_CovidPos_select_features__test_variance.csv'
    CovidPos_target = 'CovidPos'
    
    Credit_Score_train = '../../datasets/tests/7_Credit_Score_train.csv'
    Credit_Score_test = '../../datasets/tests/6_Credit_Score_select_features__test_variance.csv'
    Credit_Score_target = 'Credit_Score'

    NaiveBayes(CovidPos_train, CovidPos_test, CovidPos_target)
    NaiveBayes(CovidPos_train, CovidPos_test, CovidPos_target, 'recall')
    NaiveBayes(CovidPos_train, CovidPos_test, CovidPos_target, 'precision')
    best_model_results(CovidPos_train, CovidPos_test, CovidPos_target, 'BernoulliNB', BernoulliNB(), 'accuracy')

    NaiveBayes(Credit_Score_train, Credit_Score_test, Credit_Score_target)
    NaiveBayes(Credit_Score_train, Credit_Score_test, Credit_Score_target, 'recall')
    NaiveBayes(Credit_Score_train, Credit_Score_test, Credit_Score_target, 'precision')
    best_model_results(Credit_Score_train, Credit_Score_test, Credit_Score_target, 'BernoulliNB', BernoulliNB(), 'accuracy')
