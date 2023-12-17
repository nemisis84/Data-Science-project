from numpy import array
import pandas as pd
from pandas import Series
from matplotlib.pyplot import figure, subplots, savefig
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from helpers.dslabs_functions import HEIGHT, ts_aggregation_by, set_chart_labels, plot_multiline_chart

def boxplots(path, index, trg, gran, aggfun, titles):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    grans: list[str] = gran

    ss_1: Series = ts_aggregation_by(series, grans[0], agg_func=aggfun)
    ss_2: Series = ts_aggregation_by(series, grans[1], agg_func=aggfun)

    fig: Figure
    axs: array

    fig, axs = subplots(2, 3, figsize=(2 * HEIGHT, HEIGHT), gridspec_kw={'height_ratios': [2, 1.3]})

    set_chart_labels(axs[0, 0], title=titles[0])
    axs[0, 0].boxplot(series)
    set_chart_labels(axs[0, 1], title=titles[1])
    axs[0, 1].boxplot(ss_1)
    set_chart_labels(axs[0, 2], title=titles[2])
    axs[0, 2].boxplot(ss_2)

    axs[1, 0].grid(False)
    axs[1, 0].set_axis_off()
    axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")

    axs[1, 1].grid(False)
    axs[1, 1].set_axis_off()
    axs[1, 1].text(0.2, 0, str(ss_1.describe()), fontsize="small")

    axs[1, 2].grid(False)
    axs[1, 2].set_axis_off()
    axs[1, 2].text(0.2, 0, str(ss_2.describe()), fontsize="small")
    
    savefig(f'../../figures/data_profiling_forecast/{target}_{aggfun}_boxplots.png')

def histograms(path, index, trg, file_tag, gran, aggfun, titles):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    grans: list[str] = gran
    ss_1: Series = ts_aggregation_by(series, grans[0], agg_func=aggfun)
    ss_2: Series = ts_aggregation_by(series, grans[1], agg_func=aggfun)
    ss_3: Series = ts_aggregation_by(series, grans[2], agg_func=aggfun)
    granst: list[Series] = [series, ss_1, ss_2, ss_3]

    fig: Figure
    axs: array
    fig, axs = subplots(1, len(granst), figsize=(len(granst) * HEIGHT, HEIGHT))
    fig.suptitle(f"{file_tag} {target}")
    for i in range(len(granst)):
        set_chart_labels(axs[i], title=f"{titles[i]}", xlabel=target, ylabel="Nr records")
        axs[i].hist(granst[i].values)
    savefig(f'../../figures/data_profiling_forecast/{target}_{aggfun}_histograms.png')

def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series

def lag_plot(path, index, trg):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    figure(figsize=(3 * HEIGHT, HEIGHT))
    lags = get_lagged_series(series, 20, 10)
    plot_multiline_chart(series.index.to_list(), lags, xlabel=index, ylabel=target) 
    savefig(f'../../figures/data_profiling_forecast/{target}_logplot.png')

def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return

def correlogram(path, index, trg):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    autocorrelation_study(series, 10,1)
    savefig(f'../../figures/data_profiling_forecast/{target}_correlogram.png')


if __name__ == "__main__":
    Deaths_data = '../../datasets/forecast_covid_single.csv'
    Deaths_target = 'deaths'
    Deaths_index = 'date'

    boxplots(Deaths_data, Deaths_index, Deaths_target, ['M','Q'], 'max', ['weekly', 'monthly', 'quarterly'])
    histograms(Deaths_data, Deaths_index, Deaths_target, 'Covid', ['M','Q', 'Y'], 'max', ['weekly', 'monthly', 'quarterly', 'yearly'])
    lag_plot(Deaths_data, Deaths_index, Deaths_target)
    correlogram(Deaths_data, Deaths_index, Deaths_target)

    Total_data = '../../datasets/forecast_traffic_single.csv'
    Total_target = 'Total'
    Total_index = 'Timestamp'
    
    boxplots(Total_data, Total_index, Total_target, ['H','D'], 'sum', ['15min', 'hourly', 'daily'])
    histograms(Total_data, Total_index, Total_target, 'Traffic', ['H','D', 'W'], 'sum', ['15min', 'hourly', 'daily', 'weekly'])
    lag_plot(Total_data, Total_index, Total_target)
    correlogram(Total_data, Total_index, Total_target)