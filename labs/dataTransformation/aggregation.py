import pandas as pd
from helpers.dslabs_functions import ts_aggregation_by
import evaluate
import matplotlib.pyplot as plt

def aggregate(series, gran_level, agg_func):
    ss_agg = ts_aggregation_by(series, gran_level=gran_level, agg_func=agg_func)
    return ss_agg

if __name__ == "__main__":
    # Deaths
    # Aggregated to months
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"
    df[target] = df[target].diff().fillna(0)
    
    gran_level = "M"
    agg_func = "sum"
    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../../figures/data_transformation/1_{target}_monthly_aggregation"
    title=target+" Monthly aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
    
    # Aggregated to quarters
    gran_level = "Q"
    agg_func = "sum"
    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../../figures/data_transformation/1_{target}_quarterly_aggregation"
    title=target+" Quarterly aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
    # No aggregation

    aggregated_series = df[target]
    path=f"../../figures/data_transformation/1_{target}_no_aggregation"
    title=target+" No aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
    aggregated_series.to_csv("../../datasets/dataTransformation/1_aggregatedDeaths")

    # Total
    # Aggegated to hours
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    target = "Total"
    gran_level = "h"
    agg_func = "sum"

    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../../figures/data_transformation/1_{target}_hourly_aggregation"
    title=target+" Hourly aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
    aggregated_series.to_csv("../../datasets/dataTransformation/1_aggregatedTotal")

    # Aggregated to days
    gran_level = "d"
    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../../figures/data_transformation/1_{target}_dayly_aggregation"
    title=target+" Dayly aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
    

    # No aggregation
    aggregated_series = df[target]
    path=f"../../figures/data_transformation/1_{target}_no_aggregation"
    title=target+" No aggregation"
    evaluate.evaluateTransformation(aggregated_series, target, path=path, title=title)
