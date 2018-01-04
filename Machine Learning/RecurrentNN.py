import numpy , random , keras , pandas , sklearn

#Import data
data = pandas.read_csv('fruits.csv')																					#Import .csv dataset
X = pandas.DataFrame.as_matrix(data[['mass' , 'width', 'height' , 'color_score']])										#Convert to numpy array
examples = X.shape[0]
features = X.shape[1]
timestep = 1
X = X.reshape((examples , timestep , features))
n_featu = X.shape[2]
Y = pandas.DataFrame.as_matrix(data['fruit_label'])																		#Convert to numpy array
n_class = 4
X , Y = sklearn.utils.shuffle(X , Y , random_state = 0)																	#Shuffle the dataset to make sure similar instances are not clustering next to each other

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(3 , input_shape = (timestep , n_featu) , activation = 'relu' , return_sequences = True))
model.add(keras.layers.LSTM(3 , activation = 'relu' , dropout = 0.25 , recurrent_dropout = 0.25))
model.add(keras.layers.Dense(n_class , activation = 'softmax'))															#Output layer with 4 nodes (because of 4 classes) and the softmax activation function

#Compile model
model.compile(keras.optimizers.Adam() , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])				#Compile the model, identify here the loss function, the optimiser, and the evaluation metric

#Train model
model.fit(X , Y , batch_size = 8 , epochs = 50 , verbose = 2 , validation_split  = 0.25)								#Preform the network training, input the training feature and class tensors, identify the batch size (conventional to use multiples of 8 - 8 , 16 , 32 etc...), and the epoch number. Verbose 23 is best to printout the epoch cycle only. The validation split is splitting the dataset into a train/test set (0.25 = 25% for the test set), thus the final accuracy we want to use is the valication accuracy (where it is measured using the test set's preformace on the model)

#Prediction
PRE = numpy.array([[130 , 6.0 , 8.2 , 0.71]])
PRE = PRE.reshape((PRE.shape[0] , timestep , PRE.shape[1]))
prediction = model.predict_classes(PRE)
print(prediction)

'''
keras.layers.Dropout(rate = 0.5)																						#Percentage of nodes (of the layer after it) to be randomly switched off (their weight not updated) during an epoch, this is to prevent over fitting and allow other neurones to learn multiple features. 0.5 = 50%
keras.layers.reshape((58 , 1 , 4))																						#Reshapes a tensor into a new shape, for RNN that would be (examples , timestep of 1 , number of features)
keras.layers.LSTM(units = 2 , activation = 'relu' , return_sequences = True)											#For RNN, return_sequences = True so that the layers take into account the timesteps (learn a sequence)

The timestep is an extra dimention that allows weights from the previous epoch to be represented in the current epoch, thus datasets in sequence can be generalised
Divide the dataset into timesteps:
X = X.reshape((int(examples/timestep) , timestep , features))
					  X has 5 examples each with 4 features and a timestep of 1
					  |
					  ٧
LSTM1 (3 nodes)		(5,1,4)	<--- input_shape = (1,4) 1 timestep, and 1 weights for 1 feature therefore 4 weights
					  |
					  ٧
LSTM2 (3 nodes)		(5,1,3)	<--- 3 features because of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights. Timestep remains at 1.
					  |
					  ٧
LSTM3 (3 nodes)		(5,1,3)	<--- 3 features because of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights. Timestep remains at 1.
					  |
					  ٧
Dense4 (4 nodes)	(5,1,3)	<--- 3 features because of 3 nodes, same number of examples, therefore input_shape = (3,) 1 weights for 1 feature, therefore 3 weights. Timestep remains at 1.
					  |
					  ٧
					(5,1,4)	<--- 1 node for 1 class in Y, therefore 4 output values for 4 different classes (the probability of each class), this is compaired to the Y classes from the dataset to see if we get it right or wrong (the loss function). (examples , timestep , classes)
Divide the dataset into a timestep of 1 to be able to make a predition on an LSTM:
PRE = numpy.array([[130 , 6.0 , 8.2 , 0.71]])
PRE = PRE.reshape((PRE.shape[0] , timestep , PRE.shape[1]))
'''
