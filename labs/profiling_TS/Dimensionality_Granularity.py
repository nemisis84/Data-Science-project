import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from helpers.dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by

def most_granular(path, index, trg, filetag):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    print("Nr. Records = ", series.shape[0])
    print("First timestamp", series.index[0])
    print("Last timestamp", series.index[-1])
    plt.figure(figsize=(4 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=target,
        title=f"{filetag} {target} at the most granular detail"
    )
    plt.savefig(f'../../figures/data_profiling_forecast/{target}_most_granular.png')

def granularity(path, index, trg, filetag, gran, aggfun):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    grans: list[str] = gran
    fig: Figure
    axs: list[Axes]

    for i in range(len(grans)):

        ss_days: Series = ts_aggregation_by(series, grans[i], agg_func=aggfun)
        plt.figure(figsize=(4 * HEIGHT, HEIGHT))
        plot_line_chart(
            ss_days.index.to_list(),
            ss_days.to_list(),
            xlabel="days",
            ylabel=target,
            title=f"{filetag} {grans[i]} {aggfun} {target}",
        )
        plt.savefig(f'../../figures/data_profiling_forecast/{target}_{grans[i]}_{aggfun}_granularity.png')

if __name__ == "__main__":
    Deaths_data = '../../datasets/forecast_covid_single.csv'
    Deaths_target = 'deaths'
    Deaths_index = 'date'

    most_granular(Deaths_data, Deaths_index, Deaths_target, 'Covid')
    granularity(Deaths_data, Deaths_index, Deaths_target, 'Covid', ['M','Q', 'Y'], 'max')

    Total_data = '../../datasets/forecast_traffic_single.csv'
    Total_target = 'Total'
    Total_index = 'Timestamp'

    most_granular(Total_data, Total_index, Total_target, 'Traffic')
    granularity(Total_data, Total_index, Total_target, 'Traffic', ['H','D','W'], 'mean')