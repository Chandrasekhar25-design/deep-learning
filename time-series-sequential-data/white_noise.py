from random import gauss
from random import seed
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot

# python3

# seed random number generator
seed(1)

# create white noise series using the gauss() function
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)

# summary status
print(series.describe())
#count    1000.000000
#mean       -0.013222
#std         1.003685
#min        -2.961214
#25%        -0.684192
#50%        -0.010934
#75%         0.703915
#max         2.737260

# We can see that the mean is nearly 0.0 and the 
# standrad deviation is nearly 1.0

# line plot of the series
series.plot()
pyplot.show()
# we can see that it does appear that the series is random

# histogram plot
series.hist()
pyplot.show()
# the histogram shows the tell-tale bell-curve shape

# create a correlogram and check for any autocorrelation
autocorrelation_plot(series)
pyplot.show()
# the correlogram does not show any obvious autocorrelation pattern