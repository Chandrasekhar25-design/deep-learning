
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio

def print_2d_data(X, marker):
	plt.plot(X[:,0], X[:,1], marker)
	plt.axis('square')
	return plt

def estimate_Gaussian(X):
	m, n = X.shape
	mu = np.zeros((n,1))
	sigma2 = np.zeros((n,1))

	mu = np.mean(X, axis=0)
	sigma2 = np.var(X, axis=0)
	return mu, sigma2

def multivariate_Gaussian(X, mu, sigma2):
	k = len(mu)
	if (sigma2.shape[0]>1):
		sigma2 = np.diag(sigma2)

	X = X - mu
	argu = (2*np.pi)**(-k/2)*np.linalg.det(sigma2)**(-0.5)
	p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(sigma2))*X,axis=1))
	return p

def visulaize_fit(X, mu, sigma2):
	x = np.arange(0, 36, 0.5)
	y = np.arange(0, 36, 0.5)
	X1, X2 = np.meshgrid(x,y)
	Z = multivariate_Gaussian(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))), mu, sigma2)
	Z = Z.reshape(X1.shape)
	plt.plot(X[:,0], X[:,1], 'bx')

	if np.sum(np.isinf(Z).astype(float)) == 0:
		CS = plt.contour(X1, X2, Z, 10.**np.arange(-20, 0, 3))
		plt.clabel(CS)

	plt.show()

def select_threshold(yval, pval):
	best_epsilon = 0.
	best_F1 = 0.
	F1 = 0.
	step = (np.max(pval) - np.min(pval))/1000

	for epsilon in np.arange(np.min(pval), np.max(pval), step):
		cv_precision = pval < epsilon
		tp = np.sum((cv_precision == 1) & (yval == 1)).astype(float)
		fp = np.sum((cv_precision == 1) & (yval == 0)).astype(float)
		fn = np.sum((cv_precision == 0) & (yval == 1)).astype(float)
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		F1 = (2*precision*recall)/(precision+recall)

		if F1 > best_F1:
			best_F1 = F1
			best_epsilon = epsilon

	return best_epsilon, best_F1

def anomaly_dectection_test():
	# load and show data
	data = spio.loadmat('data1.mat')
	X = data['X']
	plt = print_2d_data(X, 'bx')
	plt.title("origin data")
	plt.show()

	mu, sigma2 = estimate_Gaussian(X)
	#print mu, sigma2

	p = multivariate_Gaussian(X, mu, sigma2)
	#print p

	visulaize_fit(X, mu, sigma2)

	Xval = data['Xval']
	yval = data['yval']  # y=1 means abnormal
	pval = multivariate_Gaussian(Xval, mu, sigma2)
	epsilon, F1 = select_threshold(yval, pval)
	print 'best epsilon: %e'%epsilon
	print 'best F1: %f'%F1

	outliers = np.where(p < epsilon)
	plt.plot(X[outliers,0],X[outliers,1],'o',markeredgecolor='r',markerfacecolor='w',markersize=10.)
	plt = print_2d_data(X, 'bx')
	plt.show()

def main():
	anomaly_dectection_test()

if __name__ == '__main__':
	main()
