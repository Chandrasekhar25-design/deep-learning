import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

np.random.seed(123) # for reproducibility

# load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print(y_train.shape)
print(y_train[:10])

# Convert 1-dimensional class arrays to 
# 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(Y_train.shape)

# declare sequential model
model = Sequential()

# declare cnn input layer
model.add(Conv2D(32, (3, 3), 
	activation='relu', input_shape=(28,28,1)))

print(model.output_shape)

model.add(Conv2D(32,(3,3), activation='relu'))
# MaxPooling2D is a way to reduce the number of 
# parameters in our model by sliding a 2x2 pooling
# filter across the previous layer and takint the max
# of the 4 values in the 2x2 filter
model.add(MaxPooling2D(pool_size=(2,2)))
# A method for regularizing our model in order 
# to prevent overfitting
model.add(Dropout(0.25)) 

# For Dense layers, the first parameter is the output
# size of the layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# the final layer has an output of 10, 10 classes of digits
model.add(Dense(10, activation='softmax'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
 
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)