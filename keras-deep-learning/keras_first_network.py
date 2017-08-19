from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, activation="relu", input_dim=8, kernel_initializer="uniform"))
model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)

# evaluate the model
#scores = model.evaluate(X,Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)