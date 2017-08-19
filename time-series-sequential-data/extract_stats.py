import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from convert_to_timeseries import covnert_data_to_timeseries

def operating_on_data():
	# input file containing data
	input_file = 'data_timeseries.txt'

	# load data
	data1 = covnert_data_to_timeseries(input_file, 2)
	data2 = covnert_data_to_timeseries(input_file, 3)
	data_frame = pd.DataFrame({'first':data1, 'second':data2})

	# plot data
	data_frame['1952':'1955'].plot()
	plt.title('Data overlapped on top of each other')

	# plot the difference
	plt.figure()
	difference = data_frame['1952':'1955']['first'] - data_frame['1952':'1955']['second']
	difference.plot()
	plt.title('Difference (first - second')

	# when 'first' is greater than a certain threshold
	# and 'second' is smaller than a certain threshold
	data_frame[(data_frame['first']>60) & (data_frame['second']<20)].plot()
	plt.title('first > 60 and second < 20')
	plt.show()


def extract_stat():
	# input file containing data
	input_file = 'data_timeseries.txt'

	# load data
	data1 = covnert_data_to_timeseries(input_file, 2)
	data2 = covnert_data_to_timeseries(input_file, 3)
	data_frame = pd.DataFrame({'first':data1, 'second':data2})

	# print max and min
	print '\nMaximum:\n', data_frame.max()
	print '\nMinimum:\n', data_frame.min()

	# print mean
	print '\nMean:\n', data_frame.mean()
	print '\nMean row-wise:\n', data_frame.mean(1)[:10]

	# plot rolling mean
	pd.rolling_mean(data_frame, window=24).plot()

	# print correlation coefficients
	print '\nCorrelation coeffiecients:\n', data_frame.corr()

	# plot rolling correlation
	plt.figure()
	pd.rolling_corr(data_frame['first'], data_frame['second'], window=60).plot()

	plt.show()

def slicing_data():
	# input file containing data
	input_file = 'data_timeseries.txt'

	# load data
	column_num = 2
	data_timeseries = covnert_data_to_timeseries(input_file, column_num)	

	# plot within a certain year range
	start = '2008'
	end = '2015'
	plt.figure()
	data_timeseries[start:end].plot()
	plt.title('Data from' + start + ' to ' + end)

	# plot within a certain range of dates
	start ='2007-2'
	end ='2007-11'
	plt.figure()
	data_timeseries[start:end].plot()
	plt.title('Data from ' + start + ' to ' + end)

	plt.show()

def main():
	#extract_stat()
	#operating_on_data()
	slicing_data()

if __name__ == '__main__':
	main()