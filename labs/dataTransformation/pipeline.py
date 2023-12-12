from aggregation import aggregate


    
def perform_aggregation(series):
    gran_level = "M"
    agg_func = "sum"
    series = aggregate(series, gran_level=gran_level, agg_func=agg_func)
    return series

def pipeline(series):
    series = perform_aggregation(series)