from aggregation import aggregate
from smoothing import smoothing
from differentiation import differentiate
from evaluate import series_train_test_split
import pandas as pd

def deaths_pipeline(series):
    # Aggregate
    gran_level = "W"
    agg_func = "sum"
    series = aggregate(series, gran_level=gran_level, agg_func=agg_func)

    # Smoothing
    method = "mean"
    param = 10
    series = smoothing(series, method, param)

    # Differentiate
    interval = 1
    series = differentiate(series, interval)

    # Final result
    train, test = series_train_test_split(series, trn_pct=0.90)
    train.to_csv("../../prepared_data/train_deaths")
    test.to_csv("../../prepared_data/test_deaths")

    return series

def total_pipeline(series):
    # Aggregate
    gran_level = "h"
    agg_func = "sum"
    series = aggregate(series, gran_level=gran_level, agg_func=agg_func)

    # Smoothing
    method = "mean"
    param = 10
    series = smoothing(series, method, param)

    # Differentiate
    interval = 7*24 # Weekly
    series = differentiate(series, interval)
    
    # Final result
    train, test = series_train_test_split(series, trn_pct=0.90)
    train.to_csv("../../prepared_data/train_total")
    test.to_csv("../../prepared_data/test_total")

if __name__ == "__main__":
    # Deaths
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"
    df[target] = df[target].diff().fillna(0)
    deaths_pipeline(df[target])

    # Total
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    target = "Total"
    total_pipeline(df[target])