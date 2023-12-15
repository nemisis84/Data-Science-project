from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from helpers.dslabs_functions import plot_line_chart, HEIGHT
from scipy.ndimage import gaussian_filter1d
import evaluate


def smoothing(data, method, param):
    if method == "mean":
        ss_smooth = data.rolling(window=param).mean()
    elif method == "median":
        ss_smooth = data.rolling(window=param).mean()
    elif method == "exp":
        ss_smooth = data.ewm(alpha=param).mean()
    ss_smooth = ss_smooth.dropna()
    return ss_smooth


if __name__=="__main__":
    # Deaths
    # Mean 10
    df = pd.read_csv('../../datasets/forecast_covid.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    target = "deaths"
    method = "mean"
    param = 10

    smoothed_series = smoothing(df[target], method, param)
    path=f"../../figures/data_transformation/2_{target}_{method+str(param)}_smoothing"
    title=target+" Monthly smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)
    
    # Median 5
    method = "median"
    param = 5

    smoothed_series = smoothing(df[target], method, param)
    path=f"../../figures/data_transformation/2_{target}_{method+str(param)}_smoothing"
    title=target+" Monthly smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)
    
    # No smoothing

    smoothed_series = df[target]
    path=f"../../figures/data_transformation/2_{target}_no_smoothing"
    title=target+" No smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)

    # Total
    # Mean 10
    df = pd.read_csv('../../datasets/forecast_traffic.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    target = "Total"
    
    method = "mean"
    param = 10

    smoothed_series = smoothing(df[target], method, param)
    path=f"../../figures/data_transformation/2_{target}_{method+str(param)}_smoothing"
    title=target+" Monthly smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)

    # Exp 0.5

    method = "exp"
    param = 0.5

    smoothed_series = smoothing(df[target], method, param)
    path=f"../../figures/data_transformation/2_{target}_{method+str(param)}_smoothing"
    title=target+" Dayly smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)

    # No smoothing
    smoothed_series = df[target]
    path=f"../../figures/data_transformation/2_{target}_no_smoothing"
    title=target+" No smoothing"
    evaluate.evaluateTransformation(smoothed_series, target, path=path, title=title)
