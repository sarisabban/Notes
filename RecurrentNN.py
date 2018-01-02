import numpy , random , keras , pandas , sklearn

#Import data
data = pandas.read_csv('fruits.csv')																					#Import .csv dataset
X = pandas.DataFrame.as_matrix(data[['mass' , 'width', 'height' , 'color_score']])										#Convert to numpy array
Y = pandas.DataFrame.as_matrix(data['fruit_label'])																		#Convert to numpy array
X , Y = sklearn.utils.shuffle(X , Y , random_state = 0)																	#Shuffle the dataset to make sure similar instances are not clustering next to each other

X_shape = (59 , 4)
X = X[: , numpy.newaxis , :]
X_shape = (59 , 1 , 4)
Y_shape = (59 , )
timestep = 1

n_class = 4

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(3 , input_shape = (1 , 4) , activation = 'relu' , return_sequences = True))
model.add(keras.layers.LSTM(3 , activation = 'relu' , dropout = 0.25 , recurrent_dropout = 0.25))
model.add(keras.layers.Dense(n_class , activation = 'softmax'))															#Output layer with 4 nodes (because of 4 classes) and the softmax activation function

#Compile model
model.compile(keras.optimizers.Adam() , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])				#Compile the model, identify here the loss function, the optimiser, and the evaluation metric

#Train model
model.fit(X , Y , batch_size = 8 , epochs = 50 , verbose = 2 , validation_split  = 0.25)								#Preform the network training, input the training feature and class tensors, identify the batch size (conventional to use multiples of 8 - 8 , 16 , 32 etc...), and the epoch number. Verbose 23 is best to printout the epoch cycle only. The validation split is splitting the dataset into a train/test set (0.25 = 25% for the test set), thus the final accuracy we want to use is the valication accuracy (where it is measured using the test set's preformace on the model)

#Prediction
PRE = numpy.array([[130 , 6.0 , 8.2 , 0.71]])
PRE = PRE[: , numpy.newaxis , :]
prediction = model.predict_classes(PRE)
print(prediction)
