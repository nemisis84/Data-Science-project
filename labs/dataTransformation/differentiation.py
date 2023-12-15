import pandas as pd
import evaluate

def differentiate(series, interval):
    ss_diff = series.diff(interval)
    ss_diff = ss_diff.dropna()
    return ss_diff

if __name__=="__main__":
    # Deaths
    # 
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"
    
    interval = 4 # Monthly

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" Monthly differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
    
    # 10
    interval = 10

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" Monthly differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
    
    # No differentiate

    differentiated_series = df[target]
    path=f"../../figures/data_transformation/3_{target}_no_differentiate"
    title=target+" No differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)

    # Total
    # hourly
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    target = "Total"
    
    interval = 4    

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" Monthly differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)

    # dayly

    interval = 4*24

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" Dayly differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)

    # No differentiate
    differentiated_series = df[target]
    path=f"../../figures/data_transformation/3_{target}_no_differentiate"
    title=target+" No differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
