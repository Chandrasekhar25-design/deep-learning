# Deep Learning with Keras

- [Keras: The python deeping learning library](https://keras.io/)

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
It was developed with a focus on enabling fast experimentation. 
_Being able to go from idea to result with the least possible delay is key to doing good research._
 
- Deep Learning

Deep learning refers to neural networks with multiple hidden layers that can learn increasingly abstract representations of the input data.
For example, deep learning has led to major advances in computer vision. We're now able to classify images, find objects in them, and even label them with captions.
To do so, deep neural networks with many hidden layers can sequentially learn more complex features from the raw input image:
  - The first hidden layers might only learn local edge patterns.
  - Then, each subsequent layer (or filter) learns more complex representations.
  - Finally, the last layers can classify the image as a cat or tiger.

These types of deep neural networks are called Convolutional Neural Networks.

- Convolutional Neural Network

In a nutshell, CNN's are multi-layer neural networks (sometimes up to 17 or more layers) that assume the input data to be images.

![](../images/typical_cnn_architecture.png)

Typical CNN Architecture, [image source](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)

By making this requirement, CNN's can efficiently handle the high dimensionality of raw images.
More information [here](http://cs231n.github.io/convolutional-networks/).

## Keras first network

- Install keras
```sh
$ pip3 install keras
Installing collected packages: numpy, scipy, six, theano, pyyaml, keras
Successfully installed keras-2.0.6 numpy-1.13.1 pyyaml-3.12 scipy-0.19.1 six-1.10.0 theano-0.9.0

$ pip3 install tensorflow
```

The steps are as follows:
1. Load Data
2. Define Model
3. Compile Model
4. Fit Model
5. Evaluate Model
6. Make predictions

### 1. Load Data

Whenever we work with machine learning algorithms that use a stochastic process (e.g. random numbers), it is a good idea to set the random number seed. This is so that you can run the same code again and get the same result.

In this tutorial, we are going to use the [Pima Indians onset of diabetes dataset](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes). It is a binary classification problem (onset of diabetes as 1 or not as 0). All of the input variables that describe each patient are numerical.

```py
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
``` 

### 2. Define Model

Models in Keras are defined as a sequence of layers. We create a Sequential model and add layers one at a time until we are happy with our network topology.

The first thing to get right is to ensure the input layer has the right number of inputs. This can be specified when creating the first layer with the **input_dim** argument and setting it to 8 for the 8 input variables.

In this example, we will use a fully-connected network structure with three layers. Fully connected layers are defined using the Dense class. We can specify the number of neurons in the layer as the first argument, the initialization method as the second argument as **init** and specify the activation function using the **activation** argument.

In this case, we initialize the network weights to a small random number generated from a **uniform** distribution ('uniform'), in this case between 0 adn 0.05 because that is the default uniform weight initialization in Keras. Another traditional alternative would be 'normal' for small random numbers generated from a **Gaussian distribution**.

We will use the rectifier(**relu**) activation function on the first two layers and the sigmoid function in the output layer. It used to be the case that sigmoid and tanh activation functions were preferred for all layers. These days, better perofrmance is achieved using the rectifier activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

```py
# create model
model = Sequential()
# the first layer has 12 neurons and expects 8 input variables
# the second hidden layer has 8 neurons
# the ouput layer has 1 neuron to predict the class
model.add(Dense(12, activation="relu", input_dim=8, kernel_initializer="uniform"))
model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

```

### 3. Compile Model

Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses the best way to represent for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.

When compiling, we must specify some additional properties required when training the network. Remember training a network means finding the best set of weights to make predictions for this problem.

We must specify the loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network and any optional metrics we would like to collect and report during training.

In this case, we will use logarithmic loss, which for a binary classification problem is defined in Keras as **binary_crossentropy**. We will also use the dfficient gradient descent algorithm **adam** for no other reason that it is an efficient default. Finally, because it is a classification problem, we will collect and report the classfication accuracy as the metric.

```py
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 4. Fit model

It is time to execute the model on some data. We can train or fit our model on our loaded data by calling the **fit()** function on the model.

The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the **nepochs** argument. We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the **batch_size** argument.

For this problem, we will run for a small number of iterations (150) and use a relatively small batch size of 10. Again, these can be chosen experimentally by trial and error.

```py
# fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)
```

### 5. Evaluate model

We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.
But ideally, you could separate your data into train and test datasets for training and evaluation of your model.

You can evaluate your model on your training dataset using the **evaluate()** function on your model and pass it the same input and output used to train the model.

This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.

```py
# evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# acc: 76.30%
```

### 6. Make predictions

We can adapt the above  example and use it to generate predictions on the training dataset, pretending it is a new dataset we have not seen before.

Making predictions is as easy calling **model.predict()**. We are using a sigmoid activation function on the output layer, so the predictions will be in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.

```py
# calculate predictions
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)
```

Let's tie it all together into a complete code example: [Keras first network](keras_first_network.py).

Running this example, you should see a message for each of the 150 epochs printing the loss and accuracy for each, followed by the final evaluation of the trained model on the training dataset or the predictions for each input pattern.

## Long Short-Term Memory Network

The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained using Backpropagation Through Time and overcomes the vanishing gradient problem.

As such, it can be used to create large recurrent networks that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results. 

Instead of neurons, LSTM networks have memory blocks that are connected through layers.

A block has components that make it smarter than a classical neuron and a memory for recent sequences. A block contains gates that manage the block's state and output. A block operates upon an input sequence and each gate within a block uses the sigmoid activation units to control whether they are triggered or not, making the change of state and addition of information flowing through the block conditional.

There are three types of gates within a unit:
- **Forget Gate**: conditionally decides what information to throw away from the block.
- **Input Gate**: conditionally decides which values from the input to update the memory state.
- **Output Gate**: conditionally decides what to output based on input and the memory of the block.

[LSTM for Regression with Time Steps](time_series_lstm_cnn.py)

- More details [here](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

## More examples

- [Keras CNN Exmaple](keras_cnn_example.py)
- [Keras Tutorial: The Ultimate Beginner's Guide to Deep Learning in Python](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
- [example models in Keras](https://github.com/fchollet/keras/tree/master/examples)
- [Stanford's computer vision class](http://cs231n.github.io/convolutional-networks/)



## Reference

- [Develop Your First Neural Network in Python With Keras Step-By-Step](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
