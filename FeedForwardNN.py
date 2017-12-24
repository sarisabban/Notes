'''
Optimisers:		Adam Gradient Descent
Loss function:		Cross-Entropy
Activiation:		ReLU
Regularisation:		Higher = more regularisation
Neural Networks:	Feed Forward , Recurrent , Convolutional , Unsupervised (GAN)
Layers:			*Input layer:	1 layer , nodes = number of features (sometimes +1 for a biase term)
			*Output layer:	1 layer , 1 node
			*Hidden layer:	1 - 2 nodes (except for convolutional networks). The starting number of nodes can be between the number of nodes in the input later and the number of nodes in the output layer
'''
import numpy , sklearn , pandas
import tensorflow as tf
from sklearn import model_selection

data = pandas.read_csv('fruits.csv')
X = data[['mass', 'width', 'height' , 'color_score']]							#Identify features
Y = data['fruit_label']											#Identify target lable(s)/value(s)
X , Y = sklearn.utils.shuffle(X , Y , random_state = 1)                    				#shuffle the dataset before the train/test split to ensure that training and testing is fair. Some datasets might be perfectly organised where all labels are sorted from top to bottom and this causes the split to be unrepresentative of the dataset.
train_x , test_x , train_y , test_y = model_selection.train_test_split(X , Y , random_state = 0)	#Split dataset into a train set (75%) and a test set (25%) [random_state: if use None then the split will change everytime the program is run. On the other hand if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing is that everytime you use 42, you will always get the same output the first time you make the split]

feature_size =	len(X.columns)					#Number of features of each instance in the dataset
n_classes =	4						#Number of classes (labels) in the dataset
batch_size =	10						#Number of examples to run through the network at the same time each cycle
n_nodes_hl1 =	3						#Hidden layer 1 has 500 nodes
n_nodes_hl2 =	3						#Hidden layer 2 has 500 nodes
n_nodes_hl3 =	3						#Hidden layer 3 has 500 nodes
x = 		tf.placeholder('float' , [None , feature_size])	#Input Features (there are feature_size number features in each example) this is the shape of our input data's matrix (the features in the dataset are float() therefore 'float' must be indicated)
y = 		tf.placeholder('int64')				#Input Label (the labels in the dataset are int() therefore 'int64' must be indicated). Must use one_hot encoding to represent the features of a tensor using many lables (unqueal shapes) whenever called someone else
def neural_network_model(data):					#Setup the neural network
	hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([feature_size, n_nodes_hl1])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}	#Weights that come out of hidden layer 1	, it is a matrix (number of features by number in nodes in HL1)		and inside it is a random number for each position (all packaged into a tensorflow variable). The biases that are added after the weights
	hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}	#Weights that come out of hidden layer 2	, it is a matrix (number of nodes in HL1 by number in nodes in HL2)	and inside it is a random number for each position (all packaged into a tensorflow variable). The biases that are added after the weights
	hidden_layer3 = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}	#Weights that come out of hidden layer 3	, it is a matrix (number of nodes in HL2 by number in nodes in HL3)	and inside it is a random number for each position (all packaged into a tensorflow variable). The biases that are added after the weights
	outputs_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3 , n_classes]))   , 'biases' : tf.Variable(tf.random_normal([n_classes]  ))}	#Weights that come out of the output layer	, it is a matrix (number of nodes in HL3 by number classes)		and inside it is a random number for each position (all packaged into a tensorflow variable). The biases that are added after the weights
	l1 = tf.add(tf.matmul(data , hidden_layer1['weights']) , hidden_layer1['biases'])									#The model for hidden layer 1: (input_data * weights) + biases
	l1 = tf.nn.relu(l1)																	#Apply activation function for hidden layer 1
	l2 = tf.add(tf.matmul(l1   , hidden_layer2['weights']) , hidden_layer2['biases'])									#The model for hidden layer 2: (input_data * weights) + biases
	l2 = tf.nn.relu(l2)																	#Apply activation function for hidden layer 2
	l3 = tf.add(tf.matmul(l2   , hidden_layer3['weights']) , hidden_layer3['biases'])									#The model for hidden layer 3: (input_data * weights) + biases
	l3 = tf.nn.relu(l3)																	#Apply activation function for hidden layer 3
	out= tf.add(tf.matmul(l3   , outputs_layer['weights']) , outputs_layer['biases'])									#The model for the output layer: (input_data * weights) + biases - no activation function
	return(out)
def train_neural_network(x , epoch):													#Setup the training of the neural network
	prediction = neural_network_model(x)												#Run the features through the neural network
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = tf.one_hot(y , n_classes)))	#Get the values using the loss function
	optimiser = tf.train.AdamOptimizer().minimize(loss)										#Minimise the loss function
	with tf.Session() as sess:													#Start computation session
		sess.run(tf.global_variables_initializer())										#Initialize all the variables in the graph
		#Train the network
		for iters in range(epoch):												#Loop through the network (a number of epoch times)
			loss_result = 0													#The initial loss function results is, of course, 0 at the start of each epoch
			i = 0
			while i < feature_size:												#Loop through batches of the dataset to pass one batch at a time through the network until the entire dataset it passed
				start = i
				end = i + batch_size
				batch_x = numpy.array(train_x[start:end])								#Batch of the features
				batch_y = numpy.array(train_y[start:end])								#Batch of the lables
				loop , c = sess.run([optimiser, loss], feed_dict = {x : batch_x , y : batch_y})				#Calculate the loss for each layer using the loss function and put it in lossresult
				loss_result += c											#Update the result of the loss for the print out by adding all the loss results of all the layers to get the total for that particular epoch
				i += batch_size												#Update i to break the while loop when the batches run through the entire dataset
			print('Epoch {:3d} of {:3d} with a loss result of {}'.format(iters + 1 , epoch , loss_result))			#Print progress
		#Evalutation
		correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(tf.one_hot(y , n_classes) , 1))							#Get the maximum value of the prediction array and the maximum value of the label array, then compair them, if they are equal = 1 if the are not equal = 0
		accuracy = tf.reduce_mean(tf.cast(correct , 'float'))									#Convert the value of correct from int() to float(), then compute the mean across the tensor, when a tensor is inserted
		print('Accuracy:' , accuracy.eval({x : test_x , y : test_y}))								#Print final accuracy

train_neural_network(x , 100)
