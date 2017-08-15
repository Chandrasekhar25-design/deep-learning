import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def read_plot_audio():
	# read the input file
	sampling_freq, audio = wavfile.read('input_read.wav')

	# print the params
	print '\nShape: ', audio.shape
	print 'Datatype: ', audio.dtype
	print 'Duration: ', round(audio.shape[0]/float(sampling_freq),3), 'seconds'

	# normalize the values
	audio = audio/(2.**15)

	# extract first 30 values for plotting
	audio = audio[:30]

	# build the time axis
	x = np.arange(0, len(audio), 1)/float(sampling_freq)

	# convert to seconds
	x *= 1000

	# plot the chopped audio signal
	plt.plot(x, audio, color='black')
	plt.xlabel('Time (ms)')
	plt.ylabel('Amplitude')
	plt.title('Audio signal')
	plt.show()

def main():
	# file where the ouput will be saved
	out_file = 'output_generated.wav'

	# specify audio parameters
	duration = 3          # seconds
	sampling_freq = 44100  # Hz
	tone_freq = 587
	min_val = -2*np.pi
	max_val = 2*np.pi

	# generate audio
	t = np.linspace(min_val, max_val, duration*sampling_freq)
	audio = np.sin(2*np.pi*tone_freq*t)

	# add some noise
	noise = 0.4 * np.random.rand(duration*sampling_freq)
	audio += noise

	# scale it to 16-bit integer values
	scaling_factor = pow(2,15) - 1
	audio_normalized = audio/np.max(np.abs(audio))
	audio_scaled = np.int16(audio_normalized*scaling_factor)

	# write to ouput file
	wavfile.write(out_file, sampling_freq, audio_scaled)

	# extract first 100 values for plotting
	audio = audio[:100]

	# build the time axis
	x = np.arange(0, len(audio), 1)/float(sampling_freq)

	# covert to seconds
	x *= 1000

	# plot the chopped audio signal
	plt.plot(x, audio, color='black')
	plt.xlabel('Time (ms)')
	plt.ylabel('Amplitude')
	plt.title('Audio signal')
	plt.show()

if __name__ == '__main__':
	main()