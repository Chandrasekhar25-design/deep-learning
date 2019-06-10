

# Deep Learning With PyTorch and TensorFlow


## Basics

- [Deep learning core concepts](resources/deep-learning-core-concepts.md).
- [Deep learning training](resources/deep-learning-training.md).
- [Understanding LSTM networks](resources/understanding_LSTM_networks.md)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
- [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) for an overview on word embeddings and RNNs for NLP
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs work specifically, but also informative about RNNs in general
- [Calculus on **Computational Graphs**](http://colah.github.io/posts/2015-08-Backprop/)

More examples:

- [apachecn/MachineLearning](https://github.com/apachecn/MachineLearning)
- [Implementation of Reinforcement Learning Algorithms. Python, OpenAI Gym, Tensorflow](https://github.com/dennybritz/reinforcement-learning)
- [lawlite19/DeepLearning_Python](https://github.com/lawlite19/DeepLearning_Python)
- [A collection of tutorials and examples for solving and understanding machine learning and pattern classification tasks](https://github.com/rasbt/pattern_classification)
- [Deep Learning papers reading roadmap for anyone who are eager to learn this amazing tech](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)
- [Content for Udacity's Machine Learning curriculum](https://github.com/udacity/machine-learning)
- [This is the lab repository of my honours degree project on machine learning](https://github.com/ShokuninSan/machine-learning)
- [A curated list of awesome Machine Learning frameworks, libraries and software](https://github.com/josephmisiti/awesome-machine-learning)
- [Bare bones Python implementations of some of the fundamental Machine Learning models and algorithms](https://github.com/eriklindernoren/ML-From-Scratch)
- [The "Python Machine Learning" book code repository and info resource](https://github.com/rasbt/python-machine-learning-book)

## Deep Learning with PyTorch

**PyTorch 0.4+** is recommended. 

### 1 - PyTorch basics

* [Offical PyTorch tutorials](http://pytorch.org/tutorials/) for more tutorials (some of these tutorials are included there)
* [PyTorch Basics](pytorch/01-basics/pytorch_basics/main.py)
* [Linear Regression](pytorch/01-basics/linear_regression/main.py#L22-L23)
* [Logistic Regression](pytorch/01-basics/logistic_regression/main.py#L33-L34)
* [Feedforward Neural Network](pytorch/01-basics/feedforward_neural_network/main.py#L37-L49)

### 2 - Intermediate
* [Convolutional Neural Network](pytorch/02-intermediate/convolutional_neural_network/main.py#L35-L56)
* [Deep Residual Network](pytorch/02-intermediate/deep_residual_network/main.py#L76-L113)
* [Recurrent Neural Network](pytorch/02-intermediate/recurrent_neural_network/main.py#L39-L58)
* [Bidirectional Recurrent Neural Network](pytorch/02-intermediate/bidirectional_recurrent_neural_network/main.py#L39-L58)
* [Language Model (RNN-LM)](pytorch/02-intermediate/language_model/main.py#L30-L50)

### 3 - Advanced
* [Image Captioning (CNN-RNN)](tutorials/03-advanced/image_captioning)
* [Deep Convolutional GAN (DCGAN)](tutorials/03-advanced/deep_convolutional_gan)
* [Variational Auto-Encoder](tutorials/03-advanced/variational_auto_encoder)
* [Neural Style Transfer](tutorials/03-advanced/neural_style_transfer)

* [Generative Adversarial Networks](pytorch/03-advanced/generative_adversarial_network/main.py#L41-L57)
* [Variational Auto-Encoder](pytorch/03-advanced/variational_autoencoder/main.py#L38-L65)
* [Neural Style Transfer](pytorch/03-advanced/neural_style_transfer)
* [Image Captioning (CNN-RNN)](pytorch/03-advanced/image_captioning)

### 4 - Utilities
* [TensorBoard in PyTorch](pytorch/04-utils/tensorboard)


### More Examples

- [spro/practical-pytorch](https://github.com/spro/practical-pytorch)
- [jcjohnson's PyTorch examples](https://github.com/jcjohnson/pytorch-examples) for a more in depth overview (including custom modules and autograd functions)
- [chenyuntc/pytorch-book](https://github.com/chenyuntc/pytorch-book)



## Deep Learning with TensorFlow

**TensorFlow v2.0**  is recommended. Added many new examples (kmeans, random forest, multi-gpu training, layers api, estimator api, dataset api ...).

### 1 - Introduction
- **Hello World** ([notebook](tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb)). Very simple example to learn how to print "hello world" using TensorFlow 2.0.
- **Basic Operations** ([notebook](tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb)). A simple example that cover TensorFlow 2.0 basic operations.

### 2 - Basic Models
- **Linear Regression** ([notebook](tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb)). Implement a Linear Regression with TensorFlow 2.0.
- **Logistic Regression** ([notebook](tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb)). Implement a Logistic Regression with TensorFlow 2.0.
- **Simple Neural Network** ([notebook](tensorflow_v2/notebooks/2_BasicModels/simple_Neural_Network.ipynb)). Implement a Simple Neural Network with TensorFlow 2.0.
- **Create your own layer** ([notebook](tensorflow_v2/notebooks/2_BasicModels/create_custom_layer.ipynb)). Create your own layer with TensorFlow 2.0.

### 3 - Neural Networks

#### Supervised
- **Simple Neural Network** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb)). Use TensorFlow 2.0 'layers' and 'model' API to build a simple neural network to classify MNIST digits dataset.
- **Simple Neural Network (low-level)** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb)). Raw implementation of a simple neural network to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb)). Use TensorFlow 2.0 'layers' and 'model' API to build a convolutional neural network to classify MNIST digits dataset.
- **Convolutional Neural Network (low-level)** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)). Raw implementation of a convolutional neural network to classify MNIST digits dataset.


#### Unsupervised
- **Auto-Encoder** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.


### 4 - Utilities
- **Save and Restore a model** ([notebook](tensorflow_v2/notebooks/4_Utils/save_restore_model.ipynb)). Save and Restore a model with TensorFlow 2.0.
- **Build Custom Layers & Modules** ([notebook](tensorflow_v2/notebooks/4_Utils/build_custom_layers.ipynb)). Learn how to build your own layers / modules and integrate them into TensorFlow 2.0 Models.


### More Examples

- [Tensorflow-2.0 Quick Start Guide](https://github.com/PacktPublishing/Tensorflow-2.0-Quick-Start-Guide)
- [aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [Official tutorial](https://github.com/tensorflow/docs/tree/master/site/en/r2/tutorials)
- [Awesome TensorFlow 2](https://github.com/Amin-Tgz/Awesome-TensorFlow-2)