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
    target = "deaths"
    gran_level = "M"
    agg_func = "sum"

    aggregated_series = aggregate(df, gran_level, agg_func)
    path=f"../..data_transformation/2_{target}_monthly_aggregation.png"
    title=target+"Monthly aggregation"
    evaluate.evaluateTransformation(aggregated_series.reset_index()[target], path=path, title=title)
    
    # No aggregation

    aggregated_series = df[target]
    path=f"../..data_transformation/2_{target}_no_aggregation.png"
    title=target+"No aggregation"
    evaluate.evaluateTransformation(aggregated_series, path=path, title=title)

    # Total
    # Aggegated to hours
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    target = "Total"
    gran_level = "h"
    agg_func = "sum"

    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../..data_transformation/2_{target}_hourly_aggregation.png"
    title=target+"Monthly aggregation"
    evaluate.evaluateTransformation(aggregated_series, path=path, title=title)

    # Aggregated to days
    gran_level = "d"
    aggregated_series = aggregate(df[target], gran_level, agg_func)
    path=f"../..data_transformation/2_{target}_dayly_aggregation.png"
    title=target+"Dayly aggregation"
    evaluate.evaluateTransformation(aggregated_series, path=path, title=title)

    # No aggregation
    aggregated_series = df[target]
    path=f"../..data_transformation/2_{target}_no_aggregation.png"
    title=target+"No aggregation"
    evaluate.evaluateTransformation(aggregated_series, path=path, title=title)
