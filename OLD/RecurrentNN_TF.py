import numpy , sklearn , pandas
import tensorflow as tf
from tensorflow.python.ops import rnn , rnn_cell
from sklearn import model_selection

#Import data
data =									pandas.read_csv('fruits.csv')
X =										data[['mass' , 'width', 'height' , 'color_score']]
Y =										data['fruit_label']
X , Y =									sklearn.utils.shuffle(X , Y , random_state = 0)
Y =										pandas.DataFrame.as_matrix(Y)								#Convert to numpy array
X =										pandas.DataFrame.as_matrix(X)								#Convert to numpy array
train_x , test_x , train_y , test_y =	model_selection.train_test_split(X , Y , random_state = 0)

#Neural network structure
n_features =	len(X[0])
n_classes =		4
batch_size =	16
chunk_size =	2
n_chunks =		2
rnn_size =		16
epoch = 		100
x = 			tf.placeholder('float' , [None , n_chunks , chunk_size])
y = 			tf.placeholder('int64')
#Setup neural network
layer				= {'weights' : tf.Variable(tf.random_normal([rnn_size , n_features])) , 'biases' : tf.Variable(tf.random_normal([n_features]))}
x1 					= tf.transpose(x , [1,0,2])
x2 					= tf.reshape(x1 , [-1 , chunk_size])
x3					= tf.split(x2 , n_chunks , 0)
lstm_cell 			= rnn_cell.BasicLSTMCell(rnn_size)
outputs , states	= rnn.static_rnn(lstm_cell , x3 , dtype = tf.float32)
out					= tf.add(tf.matmul(outputs[-1] , layer['weights']) , layer['biases'])
loss				= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out , labels = tf.one_hot(y , n_classes)))
optimiser			= tf.train.AdamOptimizer().minimize(loss)
#Setup evaluation
accuracy			= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out , 1) , tf.argmax(tf.one_hot(y , n_classes) , 1)) , 'float'))

#Train the network
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epochs in range(epoch):
		i = 0
		while i < n_features:
			start = i
			end = i + batch_size
			batch_x = numpy.array(train_x[start : end]).reshape((batch_size , n_chunks , chunk_size))
			batch_y = numpy.array(train_y[start : end])
			sess.run(optimiser , {x : batch_x , y : batch_y})
			i += batch_size
		print('Epoch {:4d} out of {}'.format(epochs , epoch))
	#Evalutate
	print('Accuracy:' , accuracy.eval({x : test_x.reshape((-1 , n_chunks , chunk_size)) , y : test_y}))
