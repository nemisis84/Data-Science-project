import pandas as pd
import evaluate

def differentiate(series, interval):
    ss_diff = series.diff(interval)
    ss_diff = ss_diff.dropna()
    return ss_diff

if __name__=="__main__":
    # Deaths
    # 
    df = pd.read_csv('../../datasets/dataTransformation/2_smoothedDeaths')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"
    
    interval = 1 # Monthly

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" One step differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
    differentiated_series.to_csv("../../datasets/dataTransformation/2_differentiatedDeaths")
    
    # 4 monthly
    interval = 4

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
    df = pd.read_csv('../../datasets/dataTransformation/2_smoothedTotal')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    target = "Total"
    
    interval = 1

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" One step differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)

    # weekly

    interval = 24*7

    differentiated_series = differentiate(df[target], interval)
    path=f"../../figures/data_transformation/3_{target}_{interval}_differentiate"
    title=target+" Weekly differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
    differentiated_series.to_csv("../../datasets/dataTransformation/2_differentiatedTotal")
    
    # No differentiate
    differentiated_series = df[target]
    path=f"../../figures/data_transformation/3_{target}_no_differentiate"
    title=target+" No differentiate"
    evaluate.evaluateTransformation(differentiated_series, target, path=path, title=title)
    
