import numpy , pandas
from sklearn import model_selection
from sklearn import neural_network
#--------------------------------------------------
''' Representation '''
data = pandas.read_csv('sonar.csv')
X = data[data.columns[0:60]]
Y = data[data.columns[60]]
X , Y = sklearn.utils.shuffle(X , Y , random_state = 1)
X_train , X_test , Y_train , Y_test = model_selection.train_test_split(X , Y , random_state = 0)
prediction = [[0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115]]
#--------------------------------------------------
''' Neural Network '''
ML = neural_network.MLPClassifier(hidden_layer_sizes = (100 , 100 , 100) , alpha = 3 , random_state = 0).fit(X_train , Y_train)
#ML = neural_network.MLPRegressor(hidden_layer_sizes = (10 , 100 , 1) , random_state = 0).fit(X_train , Y_train)
'''		     default values
hidden_layer_sizes	= (100 , )	#Number of hidden layers and number of units in each layer
activation		= 'relu'	#Activation function:				'identity' , 'logistic' , 'tanh' , 'relu'
solver			= 'adam'	#The solver for weight optimization:		'lbfgs' , 'sgd' , 'adam'
alpha			= 0.0001	#L2 Regularisation (lower = less regularisation)
batch_size		= 'auto'	#Size of minibatches for stochastic optimizers. If the solver is 'lbfgs', the classifier will not use minibatch. When set to 'auto', batch_size = min(200 , n_samples)
learning_rate		= 'constant'	#Learning rate schedule for weight updates	'constant' , 'invscaling' , 'adaptive'
learning_rate_init	= 0.001		#The initial learning rate used. It controls the step-size in updating the weights. Only used when solver = 'sgd' or 'adam'
power_t			= 0.5		#The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to 'invscaling'. Only used when solver = 'sgd'
max_iter		= 200		#Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps
shuffle			= True		#Whether to shuffle samples in each iteration. Only used when solver = 'sgd' or 'adam'
random_state		= None		#If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random
tol			= 0.0001	#Tolerance for the optimization. When the loss or score is not improving by at least tol for two consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops
verbose			= False		#Whether to print progress messages to stdout
warm_start		= False		#When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
momentum		= 0.9		#Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’
nesterovs_momentum	= True		#Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0
early_stopping		= False		#Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for two consecutive epochs. Only effective when solver=’sgd’ or ‘adam’
validation_fraction	= 0.1		#The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True
beta_1			= 0.9		#Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’
beta_2			= 0.999		#Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’
epsilon			= 0.00000001	#Value for numerical stability in adam. Only used when solver=’adam’
---------------------------------------------------------------------------------------------------------------------------------
.classes_				#Class labels for each output
.loss_					#The current loss computed with the loss function
.coefs_					#The ith element in the list represents the weight matrix corresponding to layer i
.intercepts_				#The ith element in the list represents the bias vector corresponding to layer i + 1
.n_iter_				#The number of iterations the solver has ran
.n_layers_				#Number of layers
.n_outputs_				#Number of outputs
.out_activation_			#Name of the output activation function
---------------------------------------------------------------------------------------------------------------------------------
.fit(X , Y)				#Fit the model to data matrix X and target(s) y
.get_params([deep])			#Get parameters for this estimator
.predict(X)				#Predict using the multi-layer perceptron classifier
.predict_log_proba(X)			#Return the log of probability estimates
.predict_proba(X)			#Probability estimates
.score(X , Y)				#Returns the mean accuracy on the given test data and labels
.set_params(**params)			#Set the parameters of this estimator
'''
#--------------------------------------------------
''' Evaluate '''
print(ML.score(X_train , Y_train))
print(ML.score(X_test , Y_test))
#--------------------------------------------------
''' Prediction '''
print(ML.predict(prediction))
