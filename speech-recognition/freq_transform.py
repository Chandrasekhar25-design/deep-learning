import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def main():
	# read the input file
	sampling_freq, audio = wavfile.read('input_freq.wav')

	# normalize the values
	audio = audio/(2.**15)

	# extract length
	length = len(audio)

	# apply Fourier transform
	transformed_signal = np.fft.fft(audio)
	half_length = int(np.ceil((length+1)/2.0))
	transformed_signal = np.abs(transformed_signal[0:half_length])
	transformed_signal /= float(length)
	transformed_signal **= 2

	# extract length of transformed signal
	length_signal =len(transformed_signal)

	# take care of even/odd cases
	if length%2:
		transformed_signal[1:length_signal] *= 2
	else:
		transformed_signal[1:length_signal-1] *= 2

	# extract power in dB
	power = 10 * np.log10(transformed_signal)

	# build the time axis
	x = np.arange(0, half_length, 1) * (sampling_freq/length) / 1000.0

	# plot the figure
	plt.figure()
	plt.plot(x, power, color='blue')
	plt.xlabel('Freq (kHz)')
	plt.ylabel('Power (dB)')
	plt.show()

if __name__ == '__main__':
	main()

