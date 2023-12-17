from numpy import array
import pandas as pd
from pandas import Series
from matplotlib.pyplot import figure, subplots, savefig, show, plot, legend
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from helpers.dslabs_functions import HEIGHT, ts_aggregation_by, set_chart_labels, plot_multiline_chart, plot_components, plot_line_chart

def seasonality(path, index, trg, file_tag, gran, freq: int = 96):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]

    plot_components(
    series,
    title=f"{file_tag} {gran} {target}",
    x_label=series.index.name,
    y_label=target,
    frequency = freq,
    )
    savefig(f'../../figures/data_profiling_forecast/{target}_seasonality.png')


def stationarity(path, index, trg, file_tag):
    df = pd.read_csv(path, index_col = index, parse_dates=True, infer_datetime_format=True)
    df.sort_values(by=index, inplace=True)
    target = trg
    series: Series = df[target]
    n: int = len(series)

    BINS = 10
    mean_line: list[float] = []

    for i in range(BINS):
        segment: Series = series[i * n // BINS : (i + 1) * n // BINS]
        mean_value: list[float] = [segment.mean()] * (n // BINS)
        mean_line += mean_value
    mean_line += [mean_line[-1]] * (n - len(mean_line))

    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=target,
        title=f"{file_tag} stationary study",
        name="original",
        show_stdev=True,
    )
    plot(series.index, mean_line, "r-", label="mean")
    legend()
    savefig(f'../../figures/data_profiling_forecast/{target}_stationarity.png')

if __name__ == "__main__":
    Deaths_data = '../../datasets/forecast_covid_single.csv'
    Deaths_target = 'deaths'
    Deaths_index = 'date'

    seasonality(Deaths_data, Deaths_index, Deaths_target, 'Covid', 'weekly', freq = 52)
    stationarity(Deaths_data, Deaths_index, Deaths_target, 'Covid')

    Total_data = '../../datasets/forecast_traffic_single.csv'
    Total_target = 'Total'
    Total_index = 'Timestamp'
    
    seasonality(Total_data, Total_index, Total_target, 'Traffic', '15min')
    stationarity(Total_data, Total_index, Total_target, 'Traffic')