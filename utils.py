from statsmodels.tsa.stattools import adfuller

def stationarity_test(series):
    """Returns the p-value of the Augmented Dickey-Fuller stationarity test
    on a pandas series. The null hypothesis is that there is a unit root, i.e.
    the series is not stationary. This tests for trend nonstationarity only."""
    return adfuller(series.dropna().values)[1]
