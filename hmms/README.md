

# HMM

## pattern recognition with hmm

- Patterns: recognizeable, obvious entities
- Input: raw data (image, signal, time series)
- Features: unique characteristics of patterns
- Model: assumptions on relationship btw. patterns & features

- Important: find set of features which
  - reduces amount/dimension of input data
  - still contains all information necessary to distinguish patterns

HMMs: stochastic models for temporal/serial data

i.e.: 

model gives P[pattern | features]
features: series of natural numbers

Applications:
  - speech recognition
  - bioinformatics (gene hunting)
  - fault detection in machinery
  - DOS watchdogs
  - medical signal processing

## Anomaly detection techniques

- Window based

Extract fixed length windows from training and test time series by moving one or more symbols at a time. The anomaly score for each test window is calculated using its similarity to the training windows. This similarity function can be distance measures like Euclidean, Manhattan or Correlation values, etc.

## Examples

- [hmmlearn-time-series.ipynb](hmmlearn-time-series.ipynb) 
- [single_speaker_word_recognition_with_hmm.ipynb](single_speaker_word_recognition_with_hmm.ipynb)
