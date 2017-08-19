import datetime
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


def hmm():
	# load data from input file
	input_file = 'data_hmm.txt'
	data = np.loadtxt(input_file, delimiter=',')

	# arrange data for training
	X = np.column_stack([data[:,2]])

	# create and train Gaussian HMM
	print "\nTraining HMM"
	num_components = 4
	model = GaussianHMM(n_components=num_components, covariance_type='diag', n_iter=1000)
	model.fit(X)

	# predict the hidden states of HMM
	hidden_states = model.predict(X)

	print "\nMeans and variances of hidden states:"
	for i in range(model.n_components):
		print "\nHidden state", i+1
		print "Mean =", round(model.means_[i][0], 3)
		print "Variance=", round(np.diag(model.means_[i])[0],3)

	# generate data using model
	num_samples = 1000
	samples, _ = model.sample(num_samples)
	plt.plot(np.arange(num_samples), samples[:,0], c='black')
	plt.title('Number of components = ' + str(num_components))

	plt.show()

def main():
	hmm()

if __name__ == '__main__':
	main()