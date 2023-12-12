import pandas as pd


def differentiate(series, interval):
    ss_diff = series.diff(interval)
    return ss_diff

if __name__=="__main__":
    # Deaths
    # 
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    target = "deaths"
    
    interval = 4 # Monthly

    differentiated_series = differentiate(df, interval)
    path=f"../..data_transformation/3_{target}_{interval}_differentiate.png"
    title=target+"Monthly differentiate"
    # evaluate.evaluateTransformation(differentiated_series.reset_index()[target], path=path, title=title)
    
    # 10
    interval = 10

    differentiated_series = differentiate(df, interval)
    path=f"../..data_transformation/3_{target}_{interval}_differentiate.png"
    title=target+"Monthly differentiate"
    # evaluate.evaluateTransformation(differentiated_series.reset_index()[target], path=path, title=title)
    
    # No differentiate

    differentiated_series = df[target]
    path=f"../..data_transformation/3_{target}_no_differentiate.png"
    title=target+"No differentiate"
    # evaluate.evaluateTransformation(differentiated_series, path=path, title=title)

    # Total
    # hourly
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    target = "Total"
    
    interval = 4    

    differentiated_series = differentiate(df[target], interval)
    path=f"../..data_transformation/3_{target}_{interval}_differentiate.png"
    title=target+"Monthly differentiate"
    # evaluate.evaluateTransformation(differentiated_series, path=path, title=title)

    # dayly

    interval = 4*24

    differentiated_series = differentiate(df[target], interval)
    path=f"../..data_transformation/3_{target}_{interval}_differentiate.png"
    title=target+"Dayly differentiate"
    # evaluate.evaluateTransformation(differentiated_series, path=path, title=title)

    # No differentiate
    differentiated_series = df[target]
    path=f"../..data_transformation/3_{target}_no_differentiate.png"
    title=target+"No differentiate"
    # evaluate.evaluateTransformation(differentiated_series, path=path, title=title)
