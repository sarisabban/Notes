#!/usr/bin/python3

'''
Optimisers:		Adam Gradient Descent
Loss function:		Cross-Entropy
Activiation:		ReLU for the hidden layers and softmax for the output layer (to give probability of different classes) or linear function (for regression)
			Leaky ReLU (small negative slope rather than 0) is sometimes used to allow better backpropagation if a certain weight have a too large of a negative weight that causes the node to never fire. Only used if large portion of the nodes are deing, other than that always use ReLU.
			Always end with Softmax (for single lable classifying), because it gives probability between 0 and 1
			Always end with Sigmoid (for multi lable classifying), because it does not give probability between 0 and 1
Neural Networks:	Feed Forward , Recurrent , Convolutional , Unsupervised (GAN)
Layers:			*Input layer:	1 layer , 1 node
			*Hidden layer:	1 - 2 layers (except for convolutional networks). The starting number of nodes can be between the number of nodes in the input later and the number of nodes in the output layer
			*Output layer:	1 layer , nodes = number of classes 

Learning Rate:		The steps taken in Gradient Descent to reach the global minima. low = more accurate but slower. If loss is increasing that means the Learning Rate is high.
'''
import numpy , random , keras , pandas , sklearn
#from sklearn import model_selection

#Import data
data = pandas.read_csv('fruits.csv')																					#Import .csv dataset
X = pandas.DataFrame.as_matrix(data[['mass' , 'width', 'height' , 'color_score']])										#Convert to numpy array
Y = pandas.DataFrame.as_matrix(data['fruit_label'])																		#Convert to numpy array
n_class = 4																												#Identify the number of classes
n_featu = X.shape[1]	
#Y = keras.utils.to_categorical(Y, n_class)																				#One-hot encoding. Depends on the Loss Function, some loss functions do not need encoding, others do
X , Y = sklearn.utils.shuffle(X , Y , random_state = 0)																	#Shuffle the dataset to make sure similar instances are not clustering next to each other
#train_x , test_x , train_y , test_y = model_selection.train_test_split(X , Y , random_state = 0)						#Split into train/test sets. This may not be needed because at the training function (line33) keras can automatically split the data to train and test sets

#Setup neural network
model = keras.models.Sequential()																						#Call the Sequential model object, which is a linear stack of layers
model.add(keras.layers.core.Dense(3 , input_shape = (n_featu,) , activation = 'relu'))									#Hidden layer 1 with 3 nodes and the relu activation function, but here (only for the first hidden later) we must identify the shape of the input features as (4,) because X has 4 features in each example
#model.add(keras.layers.core.Dropout(0.02))																				#Randomly not train (not use nodes) in a network (a form of regularisation to help prevent overfitting). rate = between 0 & 1 which is the rate of dropping nodes, 0.2 = 20% of the nodes will be randomly turned off (0.2 - 0.5 is usually ideal) too low = useless, too high = underlearning. This layer can be added anywhere (before the first hidden layer, between hidden layers, etc...). It is best used on larger networks rather than small ones
model.add(keras.layers.core.Dense(3 , activation = 'relu'))																#Hidden layer 2 with 3 nodes and the relu activation function
model.add(keras.layers.core.Dense(n_class , activation = 'softmax'))													#Output layer with 4 nodes (because of 4 classes) and the softmax activation function

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])			#Compile the model, identify here the loss function, the optimiser (Adam Gradient Descent with a Learning Rate of 0.01), and the evaluation metric

#Train model
model.fit(X , Y , batch_size = 8 , epochs = 50 , verbose = 2 , validation_split  = 0.25)								#Preform the network training, input the training feature and class tensors, identify the batch size (conventional to use multiples of 8 - 8 , 16 , 32 etc...), and the epoch number. Verbose 23 is best to printout the epoch cycle only. The validation split is splitting the dataset into a train/test set (0.25 = 25% for the test set), thus the final accuracy we want to use is the valication accuracy (where it is measured using the test set's preformace on the model)

#Output
#score = model.evaluate(test_x , test_y , verbose = 0)																	#Use the testing feature and class tensors to evaluate the model's accuracy,not really needed since we are using the validation argument in the train function (line 33)
#print('Test accuracy:' , score[1])
'''
#Save model weights to HDF5 - sudo pip3 install h5py
model_json = model.to_json()
with open('model.json' , 'w') as json_file:
	json_file.write(model_json)
model.save_weights('model.h5')
print('Saved model to disk')
#Load model and weights
with open('model.json' , 'r') as json_file:
	json = json_file.read()
load_model = keras.models.model_from_json(json)
load_model.load_weights('model.h5')
print('Loaded model from disk')
#Evaluate loaded model
load_model.compile(keras.optimizers.Adam() , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
score = load_model.evaluate(test_x , test_y , verbose = 0)
print('Test accuracy:' , score[1])
'''
#Prediction
prediction = model.predict_classes(numpy.array([[130 , 6.0 , 8.2 , 0.71]]))
print(prediction)

'''
keras.layers.Dense(units = 2 , activation = 'relu')																		#All nodes connected to each other

					  X has 5 examples each with 4 features
					  |
					  ٧
Dense1 (3 nodes)	(5,4)	<--- input_shape = (4,) 1 weights for 1 feature, therefore 4 weights
					  |
					  ٧
Dense2 (3 nodes)	(5,3)	<--- 3 features because output of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights
					  |
					  ٧
Dense3 (3 nodes)	(5,3)	<--- 3 features because output of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights
					  |
					  ٧
Dense4 (4 nodes)	(5,3)	<--- 3 features because output of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights
					  |
					  ٧
					(5,4)	<--- 1 node for 1 class in Y, therefore 4 output values for 4 different classes (the probability of each class), this is compaired to the Y classes from the dataset to see if we get it right or wrong (the loss function)
'''
