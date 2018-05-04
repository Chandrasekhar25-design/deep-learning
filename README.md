

# Deep Learning With PyTorch and TensorFlow


## Basics

- [Deep learning core concepts](resources/deep-learning-core-concepts.md).
- [Deep learning training](resources/deep-learning-training.md).
- [Understanding LSTM networks](resources/understanding_LSTM_networks.md)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
- [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) for an overview on word embeddings and RNNs for NLP
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs work specifically, but also informative about RNNs in general
- [Calculus on **Computational Graphs**](http://colah.github.io/posts/2015-08-Backprop/)


## Deep Learning with PyTorch

**PyTorch 0.3** is recommended. 

### 1 - PyTorch basics

* [Offical PyTorch tutorials](http://pytorch.org/tutorials/) for more tutorials (some of these tutorials are included there)
* [Deep Learning with PyTorch: A 60-minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
* [PyTorch Basics](tutorials/01-basics/pytorch_basics/main.py)
* [Linear Regression](tutorials/01-basics/linear_regression/main.py#L24-L31)
* [Logistic Regression](tutorials/01-basics/logistic_regression/main.py#L35-L42)
* [Feedforward Neural Network](tutorials/01-basics/feedforward_neural_network/main.py#L36-L47)
* [Introduction to PyTorch for former Torchies](https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb) if you are a former Lua Torch user

### 2 - Intermediate
* [Convolutional Neural Network](tutorials/02-intermediate/convolutional_neural_network/main.py#L33-L53)
* [Deep Residual Network](tutorials/02-intermediate/deep_residual_network/main.py#L67-L103)
* [Recurrent Neural Network](tutorials/02-intermediate/recurrent_neural_network/main.py#L38-L56)
* [Bidirectional Recurrent Neural Network](tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py#L38-L57)
* [Language Model (RNN-LM)](tutorials/02-intermediate/language_model/main.py#L28-L53)
* [Generative Adversarial Network](tutorials/02-intermediate/generative_adversarial_network/main.py#L34-L50)

### 3 - Advanced
* [Image Captioning (CNN-RNN)](tutorials/03-advanced/image_captioning)
* [Deep Convolutional GAN (DCGAN)](tutorials/03-advanced/deep_convolutional_gan)
* [Variational Auto-Encoder](tutorials/03-advanced/variational_auto_encoder)
* [Neural Style Transfer](tutorials/03-advanced/neural_style_transfer)

### 4 - Utilities
* [TensorBoard in PyTorch](tutorials/04-utils/tensorboard)


### More Examples

- [spro/practical-pytorch](https://github.com/spro/practical-pytorch)
- [jcjohnson's PyTorch examples](https://github.com/jcjohnson/pytorch-examples) for a more in depth overview (including custom modules and autograd functions)
- [chenyuntc/pytorch-book](https://github.com/chenyuntc/pytorch-book)



## Deep Learning with TensorFlow

**TensorFlow v1.4**  is recommended. Added many new examples (kmeans, random forest, multi-gpu training, layers api, estimator api, dataset api ...).

### 1 - Introduction
- **Hello World** ([notebook](notebooks/1_Introduction/helloworld.ipynb)) ([code](examples/1_Introduction/helloworld.py)). Very simple example to learn how to print "hello world" using TensorFlow.
- **Basic Operations** ([notebook](notebooks/1_Introduction/basic_operations.ipynb)) ([code](examples/1_Introduction/basic_operations.py)). A simple example that cover TensorFlow basic operations.

### 2 - Basic Models
- **Linear Regression** ([notebook](notebooks/2_BasicModels/linear_regression.ipynb)) ([code](examples/2_BasicModels/linear_regression.py)). Implement a Linear Regression with TensorFlow.
- **Logistic Regression** ([notebook](notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](examples/2_BasicModels/logistic_regression.py)). Implement a Logistic Regression with TensorFlow.
- **Nearest Neighbor** ([notebook](notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](examples/2_BasicModels/nearest_neighbor.py)). Implement Nearest Neighbor algorithm with TensorFlow.
- **K-Means** ([notebook](notebooks/2_BasicModels/kmeans.ipynb)) ([code](examples/2_BasicModels/kmeans.py)). Build a K-Means classifier with TensorFlow.
- **Random Forest** ([notebook](notebooks/2_BasicModels/random_forest.ipynb)) ([code](examples/2_BasicModels/random_forest.py)). Build a Random Forest classifier with TensorFlow.

### 3 - Neural Networks

#### Supervised

- **Simple Neural Network** ([notebook](notebooks/3_NeuralNetworks/neural_network_raw.ipynb)) ([code](examples/3_NeuralNetworks/neural_network_raw.py)). Build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Simple Neural Network (tf.layers/estimator api)** ([notebook](notebooks/3_NeuralNetworks/neural_network.ipynb)) ([code](examples/3_NeuralNetworks/neural_network.py)). Use TensorFlow 'layers' and 'estimator' API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)) ([code](examples/3_NeuralNetworks/convolutional_network_raw.py)). Build a convolutional neural network to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Convolutional Neural Network (tf.layers/estimator api)** ([notebook](notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](examples/3_NeuralNetworks/convolutional_network.py)). Use TensorFlow 'layers' and 'estimator' API to build a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](examples/3_NeuralNetworks/recurrent_network.py)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](examples/3_NeuralNetworks/bidirectional_rnn.py)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)) ([code](examples/3_NeuralNetworks/dynamic_rnn.py)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of different length.

#### Unsupervised
- **Auto-Encoder** ([notebook](notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](examples/3_NeuralNetworks/autoencoder.py)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **Variational Auto-Encoder** ([notebook](notebooks/3_NeuralNetworks/variational_autoencoder.ipynb)) ([code](examples/3_NeuralNetworks/variational_autoencoder.py)). Build a variational auto-encoder (VAE), to encode and generate images from noise.
- **GAN (Generative Adversarial Networks)** ([notebook](notebooks/3_NeuralNetworks/gan.ipynb)) ([code](examples/3_NeuralNetworks/gan.py)). Build a Generative Adversarial Network (GAN) to generate images from noise.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](notebooks/3_NeuralNetworks/dcgan.ipynb)) ([code](examples/3_NeuralNetworks/dcgan.py)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

### 4 - Utilities
- **Save and Restore a model** ([notebook](notebooks/4_Utils/save_restore_model.ipynb)) ([code](examples/4_Utils/save_restore_model.py)). Save and Restore a model with TensorFlow.
- **Tensorboard - Graph and loss visualization** ([notebook](notebooks/4_Utils/tensorboard_basic.ipynb)) ([code](examples/4_Utils/tensorboard_basic.py)). Use Tensorboard to visualize the computation Graph and plot the loss.
- **Tensorboard - Advanced visualization** ([notebook](notebooks/4_Utils/tensorboard_advanced.ipynb)) ([code](examples/4_Utils/tensorboard_advanced.py)). Going deeper into Tensorboard; visualize the variables, gradients, and more...

### 5 - Data Management
- **Build an image dataset** ([notebook](notebooks/5_DataManagement/build_an_image_dataset.ipynb)) ([code](examples/5_DataManagement/build_an_image_dataset.py)). Build your own images dataset with TensorFlow data queues, from image folders or a dataset file.
- **TensorFlow Dataset API** ([notebook](notebooks/5_DataManagement/tensorflow_dataset_api.ipynb)) ([code](examples/5_DataManagement/tensorflow_dataset_api.py)). Introducing TensorFlow Dataset API for optimizing the input data pipeline.

### 6 - Multi GPU
- **Basic Operations on multi-GPU** ([notebook](notebooks/6_MultiGPU/multigpu_basics.ipynb)) ([code](examples/5_MultiGPU/multigpu_basics.py)). A simple example to introduce multi-GPU in TensorFlow.
- **Train a Neural Network on multi-GPU** ([notebook](notebooks/6_MultiGPU/multigpu_cnn.ipynb)) ([code](examples/5_MultiGPU/multigpu_cnn.py)). A clear and simple TensorFlow implementation to train a convolutional neural network on multiple GPUs.

### More Examples

- [aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [buomsoo-kim/Easy-deep-learning-with-Keras](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras)
- [lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)
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
