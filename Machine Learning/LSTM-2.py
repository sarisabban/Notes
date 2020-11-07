# https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/

import math
import keras
import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense, Masking, Bidirectional
from keras.layers import TimeDistributed, RepeatVector

def Simple():
	''' Simple constant length regression or classification '''
	X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
	Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	X = X.reshape(len(X), 1, 1) # 9 examples of 1 timestep and 1 dimention
	Y = Y.reshape(len(Y), 1, 1) # 9 examples of 1 timestep and 1 dimention
	shape = X.shape
	model = Sequential()
	model.add(LSTM(5, return_sequences=True, batch_input_shape=shape))
	model.add(LSTM(5, return_sequences=True))
	model.add(LSTM(5))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()
	model.fit(X, Y, epochs=1000, verbose=0)
	predictions = model.predict(X, verbose=0)
	print([round(x, 1) for x in predictions[:, 0]])

def Variable():
	''' Variable length '''
	X = np.array((
	[10, 20, 30, 40, 50, 60, 70, 80, 90, 10],
	[11, 12, 13, 14, 15, 16, 17, 18],
	[22, 23, 24, 25, 26, 27]))
	Y = np.array((
	[10, 20, 30, 40, 50, 60, 70, 80, 90, 10],
	[11, 12, 13, 14, 15, 16, 17, 18],
	[22, 23, 24, 25, 26, 27]))
	print(X.shape)
	print(X)
	X = sequence.pad_sequences(X, value=0, padding='post', maxlen=12)
	Y = sequence.pad_sequences(Y, value=0, padding='post', maxlen=12)
	print(X.shape)
	print(X)
	X = X.reshape(3, 12, 1)
	Y = Y.reshape(3, 12, 1)
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(12, 1)))
	model.add(LSTM(5, return_sequences=True))
	model.add(LSTM(5, return_sequences=True))
	model.add(LSTM(5))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()
	model.fit(X, Y, epochs=1000, batch_size=2, verbose=0)
	Z = np.array(([10, 20, 30, 40, 50, 0, 0, 0, 0, 0, 0, 0]))
	Z = Z.reshape(12, 1)
	predictions = model.predict(Z, verbose=0)
	print([round(x, 1) for x in predictions[:, 0]])

def Bi1():
	''' Bidirectional LSTM variable length '''
	X = np.array((
	[10, 20, 30, 40, 50, 60, 70, 80, 90, 10],
	[11, 12, 13, 14, 15, 16, 17, 18],
	[22, 23, 24, 25, 26, 27]))
	Y = np.array((
	[10, 20, 30, 40, 50, 60, 70, 80, 90, 10],
	[11, 12, 13, 14, 15, 16, 17, 18],
	[22, 23, 24, 25, 26, 27]))
	print(X.shape)
	print(X)
	X = sequence.pad_sequences(X, value=0, padding='post', maxlen=12)
	Y = sequence.pad_sequences(Y, value=0, padding='post', maxlen=12)
	print(X.shape)
	print(X)
	X = X.reshape(3, 12, 1)
	Y = Y.reshape(3, 12, 1)
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(12, 1)))
	model.add(Bidirectional(LSTM(5, return_sequences=True)))
	model.add(Bidirectional(LSTM(5, return_sequences=True)))
	model.add(Bidirectional(LSTM(5)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()
	model.fit(X, Y, epochs=1000, batch_size=2, verbose=0)
	Z = np.array(([10, 20, 30, 40, 50, 0, 0, 0, 0, 0, 0, 0]))
	Z = Z.reshape(12, 1)
	predictions = model.predict(Z, verbose=0)
	print([round(x, 1) for x in predictions[:, 0]])

def Bi2():
	''' Bidirectional LSTM constant length '''
	length = 10
	limit = 10/4.0
	X = []
	Y = []
	for i in range(1000):
		XX = np.array([random() for _ in range(length)])
		YY = np.array([0 if x < limit else 1 for x in np.cumsum(XX)])
		X.append(XX)
		Y.append(YY)
	X = np.array(X)
	Y = np.array(Y)
	X = X.reshape(1000, length, 1)
	Y = Y.reshape(1000, length, 1)
	model = Sequential()
	model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=(10, 1)))
	model.add(Bidirectional(LSTM(16, return_sequences=True)))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, Y, epochs=1, batch_size=1, verbose=0)
	X = np.array([random() for _ in range(length)])
	Y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
	X = X.reshape(1, length, 1)
	Y = Y.reshape(1, length, 1)
	yhat = model.predict_classes(X, verbose=0)
	for i in range(length):
		print('Expected:', Y[0, i], 'Predicted', yhat[0, i])

def one_one():
	''' One-to-one LSTM '''
	length = 5
	X = np.array([i/float(length) for i in range(length)])
	Y = np.array([i/float(length) for i in range(length)])
	X = X.reshape(length, 1, 1)
	Y = Y.reshape(length, 1)
	model = Sequential()
	model.add(LSTM(5, input_shape=(1, 1)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	model.fit(X, Y, epochs=1000, batch_size=5, verbose=0)
	result = model.predict(X, batch_size=5, verbose=0)
	print(np.round(result, 1))
	print(Y.shape, result.shape)

def many_one():
	''' Many-to-one LSTM '''
	length = 5
	X = np.array([i/float(length) for i in range(length)])
	Y = np.array([0.3])
	X = X.reshape(1, length, 1)
	Y = Y.reshape(1, 1)
	model = Sequential()
	model.add(LSTM(5, input_shape=(5, 1)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	model.fit(X, Y, epochs=500, batch_size=1, verbose=0)
	result = model.predict(X, batch_size=1, verbose=0)
	print(result)
	print(Y.shape, result.shape)

def many_many():
	''' Many-to-many LSTM '''
	length = 5
	X = np.array([i/float(length) for i in range(length)])
	Y = np.array([i/float(length) for i in range(length)])
	X = X.reshape(1, length, 1)
	Y = Y.reshape(1, length)
	model = Sequential()
	model.add(LSTM(5, input_shape=(5, 1)))
	model.add(Dense(5))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	model.fit(X, Y, epochs=500, batch_size=1, verbose=0)
	result = model.predict(X, batch_size=1, verbose=0)
	print(np.round(result, 1))
	print(Y.shape, result.shape)

def many_many_TD():
	''' Many-to-many using TimeDistributed LSTM '''
	length = 5
	X = np.array([i/float(length) for i in range(length)])
	Y = np.array([i/float(length) for i in range(length)])
	X = X.reshape(1, length, 1)
	Y = Y.reshape(1, length, 1)
	model = Sequential()
	model.add(LSTM(5, input_shape=(5, 1), return_sequences=True))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	model.fit(X, Y, epochs=500, batch_size=1, verbose=0)
	result = model.predict(X, batch_size=1, verbose=0)
	print(np.round(result, 1))
	print(Y.shape, result.shape)

def E_D_RV():
	''' Many-to-many Encoder/Decoder LSTM with RepeatVector '''
	def random_sum_pairs(n_examples, n_numbers, largest):
		X, Y = [], []
		for i in range(n_examples):
			in_pattern = [random.randint(1,largest) for _ in range(n_numbers)]
			out_pattern = sum(in_pattern)
			X.append(in_pattern)
			Y.append(out_pattern)
		return X, Y
	def to_string(X, Y, n_numbers, largest):
		max_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
		Xstr = []
		for pattern in X:
			strp = '+'.join([str(n) for n in pattern])
			strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
			Xstr.append(strp)
		max_length = math.ceil(math.log10(n_numbers * (largest+1)))
		Ystr = []
		for pattern in Y:
			strp = str(pattern)
			strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
			Ystr.append(strp)
		return Xstr, Ystr
	def integer_encode(X, Y, alphabet):
		char_to_int = dict((c, i) for i, c in enumerate(alphabet))
		Xenc = []
		for pattern in X:
			integer_encoded = [char_to_int[char] for char in pattern]
			Xenc.append(integer_encoded)
		Yenc = []
		for pattern in Y:
			integer_encoded = [char_to_int[char] for char in pattern]
			Yenc.append(integer_encoded)
		return Xenc, Yenc
	def one_hot_encode(X, Y, max_int):
		Xenc = []
		for seq in X:
			pattern = list()
			for index in seq:
				vector = [0 for _ in range(max_int)]
				vector[index] = 1
				pattern.append(vector)
			Xenc.append(pattern)
		Yenc = []
		for seq in Y:
			pattern = list()
			for index in seq:
				vector = [0 for _ in range(max_int)]
				vector[index] = 1
				pattern.append(vector)
			Yenc.append(pattern)
		return Xenc, Yenc
	def invert(seq, alphabet):
		int_to_char = dict((i, c) for i, c in enumerate(alphabet))
		strings = list()
		for pattern in seq:
			string = int_to_char[np.argmax(pattern)]
			strings.append(string)
		return ''.join(strings)
	def generate_data(n_samples, n_numbers, largest, alphabet):
		# generate pairs
		X, Y = random_sum_pairs(n_samples, n_numbers, largest)
		# convert to strings
		X, Y = to_string(X, Y, n_numbers, largest)
		#print(X, Y)
		# integer encode
		X, Y = integer_encode(X, Y, alphabet)
		#print(X, Y)
		# one hot encode
		X, Y = one_hot_encode(X, Y, len(alphabet))
		# return as numpy arrays
		X, Y = np.array(X), np.array(Y)
		#print(X.shape, Y.shape)
		return X, Y
	random.seed(1)
	n_samples = 30000
	n_numbers = 2
	largest = 10
	alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
	n_chars = len(alphabet)
	n_in_seq_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
	n_out_seq_length = math.ceil(math.log10(n_numbers * (largest+1)))
	# define LSTM configuration
	n_batch = 10
	n_epoch = 2
	X, Y = generate_data(n_samples, n_numbers, largest, alphabet)
	# create LSTM
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
	model.add(RepeatVector(n_out_seq_length))
	model.add(LSTM(10, return_sequences=True))
	model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	model.fit(X, Y, epochs=n_epoch, batch_size=n_batch)
	# evaluate
	X, Y = generate_data(n_samples, n_numbers, largest, alphabet)
	result = model.predict(X, batch_size=n_batch, verbose=0)
	expected = [invert(x, alphabet) for x in Y]
	predicted = [invert(x, alphabet) for x in result]
	for i in range(20):
		print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))

def ST_same():
	''' Many-to-many Encoder/Decoder LSTM with Stateful '''
	''' input and output sequence are the same length '''
	from random import randint
	from numpy import array
	from numpy import argmax
	from pandas import DataFrame
	from pandas import concat
	def generate_sequence(length=25):
		return [randint(0, 99) for _ in range(length)]
	def one_hot_encode(sequence, n_unique=100):
		encoding = list()
		for value in sequence:
			vector = [0 for _ in range(n_unique)]
			vector[value] = 1
			encoding.append(vector)
		return array(encoding)
	def one_hot_decode(encoded_seq):
		return [argmax(vector) for vector in encoded_seq]
	def to_supervised(sequence, n_in, n_out):
		df = DataFrame(sequence)
		df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
		df.dropna(inplace=True)
		values = df.values
		width = sequence.shape[1]
		X = values.reshape(len(values), n_in, width)
		y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
		return X, y
	def get_data(n_in, n_out):
		# generate random sequence
		sequence = generate_sequence()
		# one hot encode
		encoded = one_hot_encode(sequence)
		# convert to X,y pairs
		X,y = to_supervised(encoded, n_in, n_out)
		return X,y
	X, y = get_data(5, 5)
#	for i in range(len(X)):
#		print(one_hot_decode(X[i]), '=>', one_hot_decode(y[i]))
	print(X.shape, y.shape)
	model = Sequential()
	model.add(LSTM(20, batch_input_shape=(7, 5, 100), return_sequences=True, stateful=True))
	model.add(TimeDistributed(Dense(100, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	for epoch in range(500):
		X,y = get_data(5, 5)
		model.fit(X, y, epochs=1, batch_size=7, verbose=2, shuffle=False)
		model.reset_states()
	# evaluate LSTM
	X,y = get_data(5, 5)
	yhat = model.predict(X, batch_size=7, verbose=0)
	for i in range(len(X)):
		print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))

def E_D_ST_diff():
	''' Many-to-many Encoder/Decoder LSTM with Stateful '''
	''' input and output sequence are different lengths '''
	from random import randint
	from numpy import array
	from numpy import argmax
	from pandas import DataFrame
	from pandas import concat
	def generate_sequence(length=25):
		return [randint(0, 99) for _ in range(length)]
	def one_hot_encode(sequence, n_unique=100):
		encoding = list()
		for value in sequence:
			vector = [0 for _ in range(n_unique)]
			vector[value] = 1
			encoding.append(vector)
		return array(encoding)
	def one_hot_decode(encoded_seq):
		return [argmax(vector) for vector in encoded_seq]
	def to_supervised(sequence, n_in, n_out):
		df = DataFrame(sequence)
		df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
		df.dropna(inplace=True)
		values = df.values
		width = sequence.shape[1]
		X = values.reshape(len(values), n_in, width)
		y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
		return X, y
	def get_data(n_in, n_out):
		# generate random sequence
		sequence = generate_sequence()
		# one hot encode
		encoded = one_hot_encode(sequence)
		# convert to X,y pairs
		X,y = to_supervised(encoded, n_in, n_out)
		return X,y
	X, y = get_data(5, 2)
#	for i in range(len(X)):
#		print(one_hot_decode(X[i]), '=>', one_hot_decode(y[i]))
	print(X.shape, y.shape)
	model = Sequential()
	model.add(LSTM(150, batch_input_shape=(21, 5, 100), stateful=True))
	model.add(RepeatVector(2))
	model.add(LSTM(150, return_sequences=True, stateful=True))
	model.add(TimeDistributed(Dense(100, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	for epoch in range(5000):
		X,y = get_data(5, 2)
		model.fit(X, y, epochs=1, verbose=2, shuffle=False)
		model.reset_states()
	# evaluate LSTM
	X,y = get_data(5, 2)
	yhat = model.predict(X, batch_size=21, verbose=0)
	for i in range(len(X)):
		print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))

#Simple()
#Variable()
#Bi1()
#Bi2()
#one_one()
#many_one()
#many_many()
#many_many_TD()
#E_D_RV()
#ST_same()
#E_D_ST_diff()
