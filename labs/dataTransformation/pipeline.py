from aggregation import aggregate
from smoothing import smoothing
from differentiation import differentiate



def deaths_pipeline(series):
    # Aggregate
    gran_level = "M"
    agg_func = "sum"
    series = aggregate(series, gran_level=gran_level, agg_func=agg_func)

    # Smoothing
    method = "mean"
    param = 10
    series = smoothing(series, method, param)

def total_pipeline(series):
    # Aggregate
    gran_level = "M"
    agg_func = "sum"
    series = aggregate(series, gran_level=gran_level, agg_func=agg_func)

    # Smoothing
    method = "mean"
    param = 10
    series = smoothing(series, method, param)

    # Differentiate
    interval = 4*24 # Dayly
    series = differentiate(series, interval)