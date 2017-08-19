
# Convert time series to supervised learning problem

Machine learning methods like deep learning can be used for time series forecasing. How to convert time series forecasting problems to supervised learning problems?

## Time series vs Supervised learning

A time series is a sequence of numbers that are ordered by a time index. This can be thought of as a list or column of ordered values.

Supervised learning is where you have input vaiables (X) and an output variable (y) and you use an algorithm to learn the mapping function from the input to the output.

```py
y = f(X)
```

The goal is to approximate the real underlying mapping so well that when you have new input data (X), you can predict the output variables (y) for that data.

Supervised learning problems can be further grouped into classification and regression problems.
- Classification A classification problem is when the output variable is a category, such as "red" and "blue" or "disease" and "no disease".
- Regression A regression problem is when the output variable is a real value, such as "dollar" or "weight". The contrived example above is a regression problem.

### Sliding window for time series data

Given a sequence of numbers for a time series dataset, we can restructure the data to look like a supervised learning problem by using previous time steps as input variables and use the next time step as the output variable.

Imagine we have a time series as follows:
```
time, measure
1, 100
2, 110
3, 108
4, 115
5, 120
```

We can restructure this time series dataset as a supervised learning problem by using the value at the previous time step to predict the value at the next time-step.
```
X, y
?, 100
100, 110
110, 108
108, 115
115, 120
120, ?
```

Here are some observations:
- the previous time step is the input(X) and the next time step is the output (y)
- the order between the observations is preserved
- we will delete the first row because no previous value can be used to predict the first value
- we don't have a known next value for the last value, it may be deleted while training the supervised model

The use of prior time steps to predict the next time step is called the **sliding window method**. In statistics and time series analysis, this is called a lag or lag method. The number of previous time steps is called the window width or size of the lag.

For more on this topic, see the post: [Time Series Forecasting as Supervised Learning](http://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

## Pandas shift() Function

A key function to help transform time series data inot a supervised learning problem is the Pandas _shift()_ function.

Given a DataFrame, the _shift()_ function can be used to create copies of columns that are pushed forward (rows of NaN values added to the front) or pulled back (rows of NaN values added to the end).

```py
from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
df['t-1'] = df['t'].shift(1)
print(df)
#   t  t-1
#0  0  NaN
#1  1  0.0
#2  2  1.0
#3  3  2.0
#4  4  3.0
#5  5  4.0
#6  6  5.0
#7  7  6.0
#8  8  7.0
#9  9  8.0
```

We can shift all the observations down by one time step by inserting one new row at the top. Because the new row has no data, we can use NaN to represent "no data".

Positive integer value means shifting the series forward one time step. Negative integer value pulls the observations up by inserting new rows at the end.

Technically, in time series forecasting terminology the current time (t) and future times (t+1, t+n) are forecast times and past observations (t-1, t+n) are used to make forecasts.

## The series_to_supervised() function

The pandas _shift()_ function would be useful tool as it would allow us to explore different framings of a time series problem with machine learning algorithms to see which might result in better performing models.

```py
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""	
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d' % (j+1, i)) for i in range(n_vars)]

	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = [x for x in range(10)]
data = series_to_supervised(values)
print(data)
```

- One-step univariate forecasting

Using lagged observations (e.g. t-1) as input variables to forecast the current time step (t). This can be done by specifying the length of the input sequence as an argument; for example:
```py
data = series_to_supervised(values, 3)
#   var1(t-3)  var1(t-2)  var1(t-1)  var1(t)
#3        0.0        1.0        2.0        3
#4        1.0        2.0        3.0        4
#5        2.0        3.0        4.0        5
#6        3.0        4.0        5.0        6
#7        4.0        5.0        6.0        7
#8        5.0        6.0        7.0        8
#9        6.0        7.0        8.0        9
```

- Multi-step or sequence forecasting

A different type of forecasting problem is using past observations to forecast a sequence of future observations.

We can frame a time series for sequence forecasting by specifying another argument. For example, we could frame a forecast problem with an input sequence of 2 past observations to forecast 2 future observations as follows:
```py
data = series_to_supervised(values, 2, 2)
#   var1(t-2)  var1(t-1)  var1(t)  var1(t+1)
#2        0.0        1.0        2        3.0
#3        1.0        2.0        3        4.0
#4        2.0        3.0        4        5.0
#5        3.0        4.0        5        6.0
#6        4.0        5.0        6        7.0
#7        5.0        6.0        7        8.0
#8        6.0        7.0        8        9.0
```

## Multivariate forecasting

Another important type of time series is called multivariate time series. This is where we may have observations of multiple different measures and an interest in forecasting one or more of them.

For example, we may have two sets of time series observations obs1 and obs2 and we wish to forecast one or both of these.

```py
raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values)
print(data)

#   var1(t-1)  var2(t-1)  var1(t)  var2(t)
#1        0.0       50.0        1       51
#2        1.0       51.0        2       52
#3        2.0       52.0        3       53
#4        3.0       53.0        4       54
#5        4.0       54.0        5       55
#6        5.0       55.0        6       56
#7        6.0       56.0        7       57
#8        7.0       57.0        8       58
#9        8.0       58.0        9       59
```

Running the example prints the new framing of the data, showing an input pattern with one time step for both variables and an output pattern of one time step for both variables. 

Depending on the speficics of the problem, the division of columns into X and Y components can be chosen arbitrarily, such as if the current observation of _var1_ was also provided as input and only _var2_ was to be predicted.

## Reference

- [How to Convert a Time Series to a Supervised Learning Problem in Python](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)