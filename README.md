# Time Series Analysis on Crypto Pairs

This repo contains some time series analysis using crypto pairs bar data from Kaggle: https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution. There are three main areas of analysis: the first is modelling daily volatility using GARCH models; the second is using MLPs to perform heteroskedastic regression across time; the third is modelling a classic statistical arbitrage strategy on this universe.

## 0. Initialisation
The repo can be cloned via `ssh` as follows. The `process_data.py` script is run once to clean the data.
```
git clone git@github.com:awh35/crypto_tsa.git
cd crypto_tsa
python3 -m venv env
source env/bin/activate
python3 process_data.py
```

## 1. GARCH Modelling 


## 2. Neural Network Heteroskedastic Regression


## 3. Crypto Statistical Arbitrage