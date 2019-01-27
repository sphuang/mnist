import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, adam
from keras.utils import np_utils
from keras.datasets import mnist

image_width = 28
image_size = image_width * image_width

def load_data(number = 10000):
	# dim of x_train is 60000 * 28 * 28
	# dim of y_train is 60000
	# dim of x_test is 10000 * 28 * 28
	# dim of y_test is 10000 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train[0: number]
	x_train = x_train.reshape(number, image_size).astype('float32')
	x_train = x_train / 255
	x_test = x_test.reshape(number, image_size).astype('float32')
	x_test = x_test / 255

	y_train = y_train[0: number]
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	return (x_train, y_train), (x_test, y_test)

# activation: 'sigmoid', 'relu'
# loss: 'mse', 'categorical_crossentropy'
# optimizer: SGD(lr=0.1), 'adam'
def process(activation = 'sigmoid', loss = 'mse', optimizer = SGD(lr = 0.1)):
	(x_train, y_train), (x_test, y_test) = load_data()

	model = Sequential()
	model.add(Dense(input_dim = image_size, units = 633, activation = activation))
	model.add(Dense(units = 633, activation = activation))
	model.add(Dense(units = 633, activation = activation))
	model.add(Dense(units = 10, activation = 'softmax'))

	model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

	model.fit(x_train, y_train, batch_size = 100, epochs = 20)

	# return [<loss>, <accuracy>]
	return model.evaluate(x_test, y_test)

def get_params(args):
	args_len = len(args)
	activation = args[1] if 2 <= args_len and 'relu' == args[1] else 'sigmoid'
	loss = args[2] if 3 <= args_len and 'categorical_crossentropy' == args[2] else 'mse'
	optimizer = args[3] if 4 <= args_len and 'adam' == args[3] else SGD(lr = 0.1)
	return (activation, loss, optimizer)

if __name__ == "__main__":
	args = sys.argv
	activation, loss, optimizer = get_params(args)
	result = process(activation, loss, optimizer)
	print result[1]
