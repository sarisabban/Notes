# Prints the matrix shapes of a network

import numpy as np
import random

m = 7				# Number of examples
nx= 8				# Number of features in each example
a = 0.01			# Learning rate
layer_1_nodes = 3
layer_2_nodes = 6
layer_3_nodes = 2

Y = np.random.randint(low=1, high=5, size=(m, 1))	# Input label
print('Y', Y.shape, '\n', Y)
X = np.random.randint(low=1, high=5, size=(m, nx))	# Input features
print('\nX', X.shape, '\n', X)

### --- FORWARD PROPAGATION --- ###
# LAYER 1
w1 = np.random.randn(nx, layer_1_nodes)				# Initialise weights with 0s (better use initialise with random numbers)
print('\nw1', w1.shape, '\n', w1)
b1 = np.zeros((1, layer_1_nodes))					# Inisialise biases with 0s (better use initialise with random numbers)
print('\nb1', b1.shape, '\n', b1)
Z1 = np.dot(X, w1) + b1								# Hypothesis function Z=X*w+b = to Z=w.T*X+b
print('\nZ1', Z1.shape, '\n', Z1)
y1 = 1/(1+np.exp(-Z1))								# Layer_1 activation output
print('\ny1', y1.shape, '\n', y1)

# LAYER 2
w2 = np.random.randn(layer_1_nodes, layer_2_nodes)	# Initialise weights with 0s (better use initialise with random numbers)
print('\nw2', w2.shape, '\n', w2)
b2 = np.zeros((1, layer_2_nodes))					# Inisialise biases with 0s (better use initialise with random numbers)
print('\nb2', b2.shape, '\n', b2)
Z2 = np.dot(y1, w2) + b2							# Hypothesis function
print('\nZ2', Z2.shape, '\n', Z2)
y2 = 1/(1+np.exp(-Z2))								# Layer_2 activation output
print('\ny2', y2.shape, '\n', y2)

# LAYER 3
w3 = np.random.randn(layer_2_nodes, layer_3_nodes)	# Initialise weights with 0s (better use initialise with random numbers)
print('\nw3', w3.shape, '\n', w3)
b3 = np.zeros((1, layer_3_nodes))					# Inisialise biases with 0s (better use initialise with random numbers)
print('\nb3', b3.shape, '\n', b3)
Z3 = np.dot(y2, w3) + b3							# Hypothesis Function
print('\nZ3', Z3.shape, '\n', Z3)
y3 = 1/(1+np.exp(-Z3))								# Layer_3 (output layer) activation output
print('\ny3', y3.shape, '\n', y3)

# COST
L = -(Y*np.log(y3) + (1-Y)*np.log(1-y3))			# Loss calculated only after output layer
print('\nL', L.shape, '\n', L)
J = (np.sum(L))/m									# Cost (average of loss over m examples)
print('\nJ =', J)

print('-----')

## --- BACK PROPAGATION --- ##
# Layer 3
dZ3 = Y-y3											# Derivative of Z of layer_3 (output layer)
print('\ndZ3', dZ3.shape, '\n', dZ3)
dw3 = np.dot(y2.T, dZ3)/m							# Derivative of w of layer_3 (output layer)
print('\ndw3', dw3.shape, '\n', dw3)
db3 = (np.sum(dZ3, axis=0, keepdims=True))/m		# Derivative of b of layer_3 (output layer) keep dimentions otherwise can become scaler rather than (1,1) matrix, and sum the 0th axis (squeez top to bottom)
print('\nbd3 =', db3.shape, '\n', db3)
### --- GRADIENT DECSCENT --- ###
w3 = np.subtract(w3, a*dw3)							# Update w for layer_3
print('\nw3', w3.shape ,'\n', w3)
b3 = b3 - a*db3										# Update b for layer_3
print('\nb3', b3.shape, '\n', b3)

## Layer 2
dy2 = (-Y/Z2)+((1-Y)/(1-Z2))						# Derivative of layer_2 activation function
print('\ndy2', dy2.shape, '\n', dy2)
dZ2 = np.dot(dZ3, w3.T) * dy2						# Derivative of Z of layer_2 (next layer weights . next layer dz) * (derivative of activeation function of this layer . z of this layer)
print('\ndZ2', dZ2.shape, '\n', dZ2)
dw2 = np.dot(y1.T, dZ2)/m							# Derivative of w of layer_2
print('\ndw2', dw2.shape, '\n', dw2)
db2 = (np.sum(dZ2, axis=0, keepdims=True))/m		# Derivative of b of layer_2
print('\nbd2 =', db2.shape, '\n', db2)
### --- GRADIENT DECSCENT --- ###
w2 = np.subtract(w2, a*dw2)							# Update w for layer_2
print('\nw2', w2.shape ,'\n', w2)
b2 = b2 - a*db2										# Update b for layer_2
print('\nb2', b2.shape, '\n', b2)

## Layer 1
dy1 = (-Y/Z1)+((1-Y)/(1-Z1))						# Derivative of layer_1 activation function 
print('\ndy1', dy1.shape, '\n', dy1)
dZ1 = np.dot(dZ2, w2.T) * dy1						# Derivative of Z of layer_1 (next layer weights . next layer dz) * (derivative of activeation function of this layer . z of this layer)
print('\ndZ1', dZ1.shape, '\n', dZ1)
dw1 = np.dot(X.T, dZ1)/m							# Derivative of w of layer_1
print('\ndw1', dw1.shape, '\n', dw1)
db1 = (np.sum(dZ1, axis=0, keepdims=True))/m								# Derivative of b of layer_1
print('\nbd1 =', db1.shape, '\n', db1)
### --- GRADIENT DECSCENT --- ###
w1 = np.subtract(w1, a*dw1)							# Update w for layer_1
print('\nw1', w1.shape ,'\n', w1)
b1 = b1 - a*db1										# Update b for layer_1
print('\nb1', b1.shape, '\n', b1)
