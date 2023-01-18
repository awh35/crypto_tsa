from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression


def rolling_2d_to_1d_arr(arr, func, window):
    T, N = arr.shape
    output = np.zeros((T, N), dtype=float)
    for t in tqdm(range(window, T)):
        roll_arr = arr[(t - window) : (t + 1), :]
        idxs_selected = ~np.any(np.isnan(roll_arr), axis=0)
        if not sum(idxs_selected):
            continue
        roll_arr = roll_arr[:, idxs_selected]
        output[t, idxs_selected] = func(roll_arr)
    output[output == 0] = np.nan
    return output


def rolling_df_to_series(df, func, window):
    df = df.copy().replace((-np.inf, np.inf), np.nan)
    idx, cols = df.index, df.columns
    output_arr = rolling_2d_to_1d_arr(df.values, func, window)
    return pd.DataFrame(output_arr, index=idx, columns=cols)


def residualise(arr, n_components, decomp_model):
    fa = decomp_model(n_components)
    factors = fa.fit_transform(arr[:-1])
    linear_model = LinearRegression()
    linear_model.fit(factors, arr[:-1])
    return arr[-1] - linear_model.predict(fa.transform(arr[-1].reshape(1, -1)))


def pca_residualise(arr, n_components):
    return residualise(arr, n_components, decomp_model=PCA)


def fa_residualise(arr, n_components):
    return residualise(arr, n_components, decomp_model=FactorAnalysis)
