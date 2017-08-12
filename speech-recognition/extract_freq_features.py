import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

def main():
	# read input sound file
	sampling_freq, audio = wavfile.read("input_freq.wav")

	#  extract MFCC and Filter bank features
	mfcc_features = mfcc(audio, sampling_freq)
	filter_bank_features = logfbank(audio, sampling_freq)

	# print parameters
	print '\nMFCC:\nNumber of windows =', mfcc_features.shape[0]
	print 'Length of each feature =', mfcc_features.shape[1]
	print '\nFilter bank: \nNumber of windows =', filter_bank_features.shape[0]
	print 'Length of each feature =', filter_bank_features.shape[1]

	# plot the features
	mfcc_features =mfcc_features.T
	plt.matshow(mfcc_features)
	plt.title('MFCC')

	filter_bank_features =  filter_bank_features.T
	plt.matshow(filter_bank_features)
	plt.title('Filter bank')
	plt.show()

if __name__ == '__main__':
	main()