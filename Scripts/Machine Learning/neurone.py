# Prints the matrix shapes of a neurone

import numpy as np
import random

m = 7				# Number of examples
nx= 8				# Number of features in each example
a = 0.01			# Learning rate

Y = np.random.randint(low=1, high=5, size=(m, 1))	# Input label
print('Y', Y.shape, '\n', Y)
X = np.random.randint(low=1, high=5, size=(m, nx))	# Input features
print('\nX', X.shape, '\n', X)
w = np.zeros((nx, 1))								# Initialise weights with 0s (better use initialise with random numbers)
print('\nw', w.shape, '\n', w)
b = 0												# Inisialise biase with 0s (better use initialise with random numbers)
print('\nb=', b)

### --- FORWARD PROPAGATION --- ###
Z = np.dot(X, w)+b									# Hypothesis
print('\nZ', Z.shape, '\n', Z)
y = 1/(1+np.exp(-Z))								# Activation function
print('\ny', y.shape, '\n', y)
L = -(Y*np.log(y) + (1-Y)*np.log(1-y))				# Loss
print('\nL', L.shape, '\n', L)
J = (np.sum(L))/m									# Cost
print('\nJ =', J)

## --- BACK PROPAGATION --- ##
dZ = Y-y											# Derivative of hypothesis
print('\ndZ', dZ.shape, '\n', dZ)
print('\nX.T', X.T.shape, '\n', X.T)
dw = np.dot(X.T, dZ)/m								# Derivative of weights
print('\ndW', dw.shape, '\n', dw)
db = (np.sum(dZ))/m									# Derivative of biase
print('\ndb =', db)

### --- GRADIENT DECSCENT --- ###
w = np.subtract(w, a*dw)							# Update w
print('\nw', w.shape, '\n', w)
b = b - a*db										# Update b
print('\nb =', b)
