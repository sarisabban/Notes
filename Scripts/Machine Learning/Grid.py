'''
Hyperparameter Optimisation
===========================

# Neural Network Architechture:
	* Number of layers:					Add layers until test
										loss stops improving
	* Nodes per layer:					Increase units to increase accuracy
	* Regularisation:					Add dropout 20%-50% works best with
										larger networks. weight constraint
										is a trigger that checks the size or
										magnitude of the weights and scales them
										so that they are all below a pre-defined
										threshold. The constraint forces weights
										to be small and can be used instead of
										weight decay and in conjunction with
										more aggressive network configurations,
										such as very large learning rates
										because large weights in a neural
										network are a sign of overfitting
	* Network weight initialisation:	"Uniform Distribution"
	* Activation function:				Use "ReLu" or "Leaky ReLu" between
										layers
										Use "Sigmoid" at output layer for
										binary predictions
										Use "SoftMax" at output layer for
										multiple class predictions
# Training:
	* Learning rate:	low = slow but fine, high = fast but course
						Using a decaying learning rate is preffeared
						Use momentum 0.5-0.9 prevents oscillation by knowing
						what is the next step from the previous step
	* Optimiser:		"Adam"
	* Number of epochs:	Increase until test loss get worse even if training loss
						is betting better (overfitting)
	* Batch size:		Start at 32 and increase as needed 64, 128, 256 ...
						LSTM, RNN, CNN are sensitive to batch size
						The batch size is the number of patterns shown to the
						network before the weights are updated. How many pattern
						to read at a time and keep in memory

The Grid Search Method
----------------------

Using scikit-learn's grid search to fine tune a keras deep learning model.

1. Use Keras models in scikit-learn
2. Use grid search in scikit-learn
3. Tune batch size and epochs
4. Tune optimization algorithms
5. Tune learning rate and momentum
6. Tune network weight initialization
7. Tune activation functions
8. Tune dropout regularization
9. Tune the number of nodes in a layer
'''

import os
import keras
import sklearn
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

# wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# 1. Use keras models in scikit-learn:
#	To pass a keras model into scikit-lear it must be wraped the sequential
#	model in a function then pass that function to the KerasClassifier or
#	KerasRegressor.
def create_model():
	NN = Sequential()
	NN.add(Dense(12, input_dim=8, activation='relu'))
	NN.add(Dense(8, activation='relu'))
	NN.add(Dense(1, activation='sigmoid'))
	NN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return(NN)
model = KerasClassifier(build_fn=create_model)
#	KerasClassifier class can take default arguments that are passed on to the
#	calls to model.fit(), such as the number of epochs and the batch size.
model = KerasClassifier(build_fn=create_model, epochs=10)
#	The constructor for the KerasClassifier class can also take new arguments
#	that can be passed to your custom create_model() function. These new
#	arguments must also be defined in the signature of your create_model() 
#	function with default parameters.
def create_model2(dropout_rate=0.0):
	NN = Sequential()
	NN.add(Dense(12, input_dim=8, activation='relu'))
	NN.add(Dense(8, activation='relu'))
	NN.add(Dense(1, activation='sigmoid'))
	NN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return(NN)
model = KerasClassifier(build_fn=create_model2, dropout_rate=0.2)

# 2. Use grid search in scikit-learn:
#	Grid search is a model hyperparameter optimization technique in scikit-learn
#	this technique is provided in the GridSearchCV class which is found under
#	sklearn.model_selection.GridSearchCV() when constructing this class you must
#	provide a dictionary of hyperparameters to evaluate in the param_grid an
#	argument. This is a map of the model parameter name and an array of values
#	to try.
param = dict(epochs=[10, 20, 30])
#	By default the accuracy is what will be optamised, but other scores can be
#	chosen in the score argument of the GridSearchCV(). Also by default the
#	search uses just one CPU core, this can be adjusted by passing -1 to the
#	n_jobs argument to use all available CPU cores.
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
#	The GridSearchCV process will construct and evaluate one model for each
#	combination of parameters. Cross validation is used to evaluate each
#	individual model and the default of 3-fold cross validation is used.
grid_result = grid.fit(X, Y)
#	The best_score_ member provides access to the best score observed during the
#	optimization procedure and the best_params_ describes the combination of
#	parameters that achieved the best results
BestScore = grid_result.best_score_
BestParams= grid_result.best_params_
print('Best score: {} using {}'.format(BestScore, BestParams))

# 3. Tune batch size and epochs:
#	Make a list of the values to loop through
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
#	Add these lists to a dictionary as before
param = dict(batch_size=batch_size, epochs=epochs)
#	Pass this dictionary to GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)
#	Summarize results
BestScore = grid_result.best_score_
BestParams= grid_result.best_params_
print('Best: {} using {}'.format(BestScore, BestParams))
#	Get the breakdown
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print('mean: {} ±{} with: {}'.format(mean, stdev, param))

# 4. Tune optimization algorithms:
#	Again make a list of optimisers, add them to the param dictionary and pass
#	the dictionary to GridSearchCV, but remember to pass in the optimiser
#	argument to the model function since it is not included
def create_model(optimizer='adam'):
	NN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return(NN)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

# 5. Tune learning rate and momentum:
#	similar concept to tuning the optimiser
def create_model(learn_ratern=0.01, momentum=0):
	return(NN)
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

# 6. Tune network weight initialization:
#	Ideally, it is better to use different weight initialization schemes
#	according to the activation function used on each layer, in other words
#	repeat this process for the different weight initialization on different
#	layers
def create_model(init_mode='uniform'):
	model.add(Dense(12, input_dim=8, kernel_initializer=init_mode, activation='relu'))
	return(NN)
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

# 7. Tune activation functions:
#	The activation function controls the non-linearity of individual neurons
#	and when to fire. ReLu and leaky ReLu are the most popular, but here we can
#	search for ones that better fit our neural network and dataset
def create_model(activation='relu'):
	model.add(Dense(12, input_dim=8, activation=activation))
	return(NN)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

# 8. Tune dropout regularization:
#	To get good results, dropout is best combined with a weight constraint such
#	as the max norm constraint
def create_model(dropout_rate=0.0, weight_constraint=0):
	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	return(NN)
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

# 9. Tune the number of nodes in a layer:
#	Just like the previous concepts, tune the nodes per 1 layer, repeat for each
#	layer, or combine to search all layers at the same time.
def create_model(neurons=1):
	model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
	return(NN)
neurons = [1, 5, 10, 15, 20, 25, 30]
param = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1)
grid_result = grid.fit(X, Y)

'''
TIPS:
-----
* k-fold Cross Validation:
	You can see that the results from the examples in this post show some
	variance. A default cross-validation of 3 was used, but perhaps k=5 or k=10
	would be more stable. Carefully choose your cross validation configuration
	to ensure your results are stable. model_selection.GridSearchCV(cv=10)
* Review the Whole Grid:
	Do not just focus on the best result, review the whole grid of results and
	look for trends to support configuration decisions.
* Parallelize:
	Use all your cores if you can, neural networks are slow to train and we
	often want to try a lot of different parameters. Consider spinning up a
	lot of AWS instances.
* Use a Sample of Your Dataset:
	Because networks are slow to train, try training them on a smaller sample
	of your training dataset, just to get an idea of general directions of
	parameters rather than optimal configurations.
* Start with Coarse Grids:
	Start with coarse-grained grids and zoom into finer grained grids once you
	can narrow the scope.
* Do not Transfer Results:
	Results are generally problem specific. Try to avoid favorite configurations
	on each new problem that you see. It is unlikely that optimal results you
	discover on one problem will transfer to your next project. Instead look
	for broader trends like number of layers or relationships between parameters.
* Reproducibility is a Problem:
	Although we set the seed for the random number generator in NumPy, the
	results are not 100% reproducible. There is more to reproducibility when
	grid searching wrapped Keras models than is presented in this post.
'''
