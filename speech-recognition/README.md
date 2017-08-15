# Speech Recognition

## [python_speech_features](https://github.com/jameslyons/python_speech_features)

### Installation

To install from pypi:
```sh
$ sudo pip install python_speech_features
```

### [Usage](http://python-speech-features.readthedocs.io/en/latest/)

- python_speech_features.mfcc() - Mel Frequency Cepstral Coefficients
- python_speech_features.fbank() - Filterbank Energies
- python_speech_features.logfbank() - Log Filterbank Energies
- python_speech_features.ssc() - Spectral Subband Centroids

The Mel scale perceived frequency, or pitch, of a pure tone to 
its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they
are at high frequencies. Incorporating this scale makes our features match more closely what humans hear. [More details](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)

To use MFCC features
```python
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("file.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])
```

### Example
- [Generate audio file](generate_audio.py)
- [Synthesize music](synthesize_music.py)
- [Extract freqency Features](extract_freq_features.py)
- [Extract power in dB](freq_transform.py)

## [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)
- Simple algorithms and models to learn HMM (Hidden Markov Model) in Python
- Follow scikit-learn API as close as possible, but adapted to sequence data
- Built on scikit-learn, Numpy, SciPy, and matplotlib

### [Install](https://github.com/hmmlearn/hmmlearn)

```sh
$ sudo pip install -U --user hmmlearn
```

### Examples
- [Speech recognizer](speech_recognizer.py)


## Reference
- [Python-Machine-Learning-Cookbook](https://github.com/PacktPublishing/Python-Machine-Learning-Cookbook)
