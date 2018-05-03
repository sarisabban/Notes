#!/usr/bin/python3

import keras , pandas , sklearn
from sklearn import model_selection

#Study
'''
Optimisers:		Adam Gradient Descent
Loss function:		Cross-Entropy (if the labels are integers use sparse_categorical_crossentropy if they are one-hot encoded use categorical_crossentropy)
Activiation:		ReLU for the hidden layers and softmax for the output layer (to give probability of different classes) or linear function (for regression)
			Leaky ReLU (small negative slope rather than 0) is sometimes used to allow better backpropagation if a certain weight have a too large of a negative weight that causes the node to never fire. Only used if large portion of the nodes are deing, other than that always use ReLU.
			Always end with Softmax (for single lable classifying), because it gives probability between 0 and 1
			Always end with Sigmoid (for multi lable classifying), because it does not give probability between 0 and 1
Neural Networks:	Feed Forward (FFNN), Recurrent (RNN & LSTM), Convolutional (CNN), Unsupervised (GAN)
Layers:			*First layer:	1 layer, nodes = ?. The first layer also contains the input layer defined as input_shape = (number of features,) and the ,) indicates that it is a single element tuple
			*Hidden layer:	1 - 2 layers (except for convolutional networks), nodes = ?. The starting number of nodes can be between the number of nodes in the input later and the number of nodes in the output layer. But they can increase to the 100 until good accuracy is acheived
			*Output layer:	1 layer, nodes = number of classes 
Learning Rate:		The steps taken in Gradient Descent to reach the global minima. low = more accurate but slower. If loss is increasing that means the Learning Rate is high. Default is always 0.01
'''

#To run computation on a GPU
'''
1. Install CUDA: sudo pacman -S cuda
2. Install NUMBA: pip3 install numba
3. from numba import vectorize , cuda
4. @Vectorize(['float32(float32 , float32)'] , target = cuda)
'''

#Tensor Board - sudo pacman -S tensorboard
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')
'''
Important parameters to view:
	1. Loss over epoch
	2. Accuracy over epoch

In python add this line:
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')
and add callbacks = [tensorboard] to model.fit

In the terminal execute this command, then open the local URL:
tensorboard --logdir=./logs
'''
#Import data
data = pandas.read_csv('../OLD/MNIST.csv')
X = (data.ix[:,1:].values) / 255		# Divide each vector value by 255, MinMax regularisation but for each item separatly, therefore 0-255 values become 0-1 values
Y = data.ix[:,0].values
X , Y = sklearn.utils.shuffle(X , Y , random_state = 0)
n_class = 10
n_featu = X.shape[1]
Y = keras.utils.to_categorical(Y , n_class)	# One-hot encoding
x_train , x_test , y_train , y_test = model_selection.train_test_split(X , Y , random_state = 0)

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.core.Dense(30 , input_shape = (n_featu,) , activation = 'relu'))
model.add(keras.layers.core.Dropout(0.2))
model.add(keras.layers.core.Dense(30 , activation = 'relu'))
model.add(keras.layers.core.Dropout(0.2))
model.add(keras.layers.core.Dense(n_class , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.001) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model
model.fit(x_train , y_train , batch_size = 128 , epochs = 20 , verbose = 2 , validation_data = (x_test, y_test) , callbacks = [tensorboard])
'''
#Save model weights to HDF5 - sudo pacman -S python-h5py
import h5py

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

#Prediction
#prediction = model.predict_classes(numpy.array([[130 , 6.0 , 8.2 , 0.71]]))
print(prediction)
'''

#The Shape of data
'''			Dataset Shape
			(5 , 4)						<-- 5 examples each having 4 features
			   |
			   ٧
			Layer Output Shape	Number of Parameters	<-- Number of trainable wegihts (weights coming from previous layer + 1 bias weight for each node)
input_shape:		(4    ,  )					<-- 4 is the number of features in the dataset for each example
Dense 1:	3 Nodes	(None , 3)		15			<-- Dense layer 1 and the input_shape are coded in the same line in KERAS
Dropout 1:		//			0			<-- Randomly freez 20% of the nodes every epoch to reduce overfitting and get better generalisation
Dense 2:	3 Nodes	(None , 3)		12			<-- None refers to any value in this position , 3 for 3 nodes
Dropout 2:		//			0			<-- Best to use dropout layers between each layer
Dense 3:	2 Nodes	(None , 2)		8			<-- 2 for 2 output nodes (each node for each class)
'''
