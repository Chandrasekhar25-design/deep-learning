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

- [Extract freqency Features](extract_freq_features.py)
