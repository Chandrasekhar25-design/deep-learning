
# Time Series and Sequential Data

## Install
- [Pandas](http://pandas.pydata.org/getpandas.html)
- [PyStruct](https://pystruct.github.io/installation.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

## White noise
- A time series is white noise if the variables are independent and identically distributed with a mean of zero.
- This means that all variables have the same variance (_sigma_^2) and each value has a zero correlcation with all other values in the series.
- If a time series is white noise, it is sequence of random numbers and cannot be predicted.
- If the series of forecast errors are not white noise, it suggests improvements could be made to the predictive model.
- [Example of white noise time series](white_noise.py)
- [More details](http://machinelearningmastery.com/white-noise-time-series-python/)

## Convert to time series
- [Read data and convert to time series](convert_to_timeseries.py)
- [Extract statistic, operating on data and slicing data](extract_stats.py)

## Convert time series to supervised learning
- [Time series to supervised learning](time-series-to-supervised-learning.md)

## HMM
- [hmm](hmm.y)
