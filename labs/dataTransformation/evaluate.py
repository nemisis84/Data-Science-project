import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from helpers.dslabs_functions import PAST_COLOR, FUTURE_COLOR, PRED_PAST_COLOR, PRED_FUTURE_COLOR, HEIGHT, FORECAST_MEASURES
from helpers.dslabs_functions import plot_multibar_chart
from sklearn.linear_model import LinearRegression

def series_train_test_split(data, trn_pct: float = 0.90):
    trn_size: int = int(len(data) * trn_pct)
    df_cp = data.copy()
    train = df_cp.iloc[:trn_size]
    test = df_cp.iloc[trn_size:]
    return train, test

def plot_forecasting_series(
    trn,
    tst,
    prd_tst,
    title: str = "",
    xlabel: str = "time",
    ylabel: str = "",):

    fig, ax = plt.subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT), squeeze=True)
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(trn.index, trn.values, label="train", color=PAST_COLOR)
    ax.plot(tst.index, tst.values, label="test", color=FUTURE_COLOR)
    ax.plot(prd_tst.index, prd_tst.values, "--", label="test prediction", color=PRED_FUTURE_COLOR)
    ax.legend(prop={"size": 5})

    return ax

def plot_dual_forecasting_series(plot1, plot2, path=False):
    # Set up a figure with two subplots, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the figsize as needed
    for line in plot1.lines:
        ax1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())
    
    # Transfer the lines from plot2 to ax2
    for line in plot2.lines:
        ax2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())


    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_forecasting_eval(trn, tst, prd_trn, prd_tst, title: str = ""):
    ev1: dict = {
        "RMSE": [sqrt(FORECAST_MEASURES["MSE"](trn, prd_trn)), sqrt(FORECAST_MEASURES["MSE"](tst, prd_tst))],
        "MAE": [FORECAST_MEASURES["MAE"](trn, prd_trn), FORECAST_MEASURES["MAE"](tst, prd_tst)],
    }
    ev2: dict = {
        "MAPE": [FORECAST_MEASURES["MAPE"](trn, prd_trn), FORECAST_MEASURES["MAPE"](tst, prd_tst)],
        "R2": [FORECAST_MEASURES["R2"](trn, prd_trn), FORECAST_MEASURES["R2"](tst, prd_tst)],
    }

    # print(eval1, eval2)
    fig, axs = plt.subplots(1, 2, figsize=(1.5 * HEIGHT, 0.75 * HEIGHT), squeeze=True)
    fig.suptitle(title)
    plot_multibar_chart(["train", "test"], ev1, ax=axs[0], title="Scale-dependent error", percentage=False)
    plot_multibar_chart(["train", "test"], ev2, ax=axs[1], title="Percentage error", percentage=True)
    return axs

def plot_dual_forecasting_eval(plot1, plot2, path=False):
    # Set up a figure with two subplots, one for each call to plot_forecasting_eval
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Adjust figsize as needed

    # Move the content of the axes returned by your method to the first subplot
    for item in plot1.get_children():
        ax1.add_artist(item)

    # Move the content of the axes returned by your method to the second subplot
    for item in plot2.get_children():
        ax2.add_artist(item)
    
    # Adjust the layout
    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()


def evaluateTransformation(data, target, path=False, title = ""):
    train, test = series_train_test_split(data, trn_pct=0.90)
    trnX = np.arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = np.arange(len(train), len(data)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn = pd.Series(model.predict(trnX), index=train.index)
    prd_tst  = pd.Series(model.predict(tstX), index=test.index)
    plt.figure()
    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Evaluation {title}")
    if path:
        plt.savefig(path + ".png")
    else:
        plt.show()
    plt.figure()
    plot_forecasting_series(train, test, prd_tst, title=f"Forecasting {title}", xlabel=data.index, ylabel=target)
    if path:
        plt.savefig(path + "_forecasting.png")
    else:
        plt.show()



if __name__ == "__main__":
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df.drop(columns=["week"], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"

    data = df[target]
    evaluateTransformation(data, title="Covid after aggregation")

