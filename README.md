# Time Series Analysis on Crypto Pairs

This repository contains some time series analysis using crypto pairs bar data from [Kaggle](https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution). Naturally there is heteroskedasticity in this data. An obvious solution to this would be to manually normalise the returns series across time, but this repository focuses on a few approaches which explicitly model the heteroskedasticity. It also contains an implementation of a classical statistical arbitrage strategy on these pairs.

## 0. Initialisation
The repo can be cloned via `ssh` as follows. The `process_data.py` script is run once to clean the data.
```
git clone git@github.com:awh35/crypto_tsa.git
cd crypto_tsa
python3 -m venv env
source env/bin/activate
python3 process_data.py
```

## 1. Crypto Statistical Arbitrage
This notebook fits PCA and Factor Analysis on a daily basis, and computes out-of-sample residuals by regressing each crypto pair onto these latent factors. It also fits the Ornstein-Uhlenbeck (OU) process to these residuals. It then uses these (possibly OU-preprocessed) residuals to simulate a trading strategy on a small universe of crypto pairs.


## 2. GARCH Modelling 
This notebook contains a simple implementation of the classical GARCH model for heteroskedastic returns series. It is evaluated statistically across time. 


## 3. Neural Network Heteroskedastic Regression
This notebook contains a simple implementation of an MLP for heteroskedastic regression. It achieves this by simultaneously predicting both the mean and variance of the targets. Due to compute constraints, this is trained as a one-off, and no historical evaluation is performed.
