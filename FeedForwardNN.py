import numpy , sklearn , pandas
import tensorflow as tf
from sklearn import model_selection

data = pandas.read_csv('fruits.csv')
X = data[['mass', 'width', 'height' , 'color_score']]
Y = data['fruit_label']
X , Y = sklearn.utils.shuffle(X , Y , random_state = 1)
train_x , test_x , train_y , test_y = model_selection.train_test_split(X , Y , random_state = 0)

n_features =	len(X.columns)
n_classes =	4
batch_size =	10
n_nodes_hl1 =	3
n_nodes_hl2 =	3
n_nodes_hl3 =	3
epoch = 	10
x = 		tf.placeholder('float' , [None , n_features])
y = 		tf.placeholder('int64')

#Setup neural network
hidden_layer1	= {'weights' : tf.Variable(tf.random_normal([n_features  , n_nodes_hl1])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_layer2	= {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden_layer3	= {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
outputs_layer	= {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3 , n_classes]))   , 'biases' : tf.Variable(tf.random_normal([n_classes]  ))}
l1		= tf.nn.relu(tf.add(tf.matmul(x    , hidden_layer1['weights']) , hidden_layer1['biases']))
l2		= tf.nn.relu(tf.add(tf.matmul(l1   , hidden_layer2['weights']) , hidden_layer2['biases']))
l3		= tf.nn.relu(tf.add(tf.matmul(l2   , hidden_layer3['weights']) , hidden_layer3['biases']))
out		=            tf.add(tf.matmul(l3   , outputs_layer['weights']) , outputs_layer['biases'])
loss		= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out , labels = tf.one_hot(y , n_classes)))
optimiser	= tf.train.AdamOptimizer().minimize(loss)

#Train the network
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epochs in range(epoch):
		i = 0
		while i < n_features:
			start = i
			end = i + batch_size
			batch_x = numpy.array(train_x[start : end])
			batch_y = numpy.array(train_y[start : end])
			sess.run(optimiser , {x : batch_x , y : batch_y})
			i += batch_size
		print('Epoch {:4d} out of {}'.format(epochs , epoch))




	#Evalutation
	correct = tf.equal(tf.argmax(out , 1) , tf.argmax(tf.one_hot(y , n_classes) , 1))	#Get the maximum value of the prediction array and the maximum value of the label array, then compair them, if they are equal = 1 if the are not equal = 0
	accuracy = tf.reduce_mean(tf.cast(correct , 'float'))					#Convert the value of correct from int() to float(), then compute the mean across the tensor, when a tensor is inserted
	print('Accuracy:' , accuracy.eval({x : test_x , y : test_y}))
