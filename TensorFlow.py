#pip3 install tensorflow
import tensorflow , numpy , pandas , matplotlib.pyplot , sklearn.utils
from sklearn import model_selection

#Read data - taken from https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
df = pandas.read_csv('sonar.csv')
X = df[df.columns[0:60]].values	#Use .values to put values in an array
Y = df[df.columns[60]]		#0 for Rock , 1 for Mine
#Shuffle Dataset
X , Y = sklearn.utils.shuffle(X , Y , random_state = 1)
#Split Dataset into train and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X , Y , random_state = 0)

#TensorFlow - Graph
learning_rate	= 0.03						#
training_epoch	= 1000						#Number it iterations to minimise error
cost_history	= numpy.empty(shape = [1] , dtype = float)	#
n_dim		= X.shape[1]					# Number of features in the X dataset
n_class		= 2						# Number of class labels in the Y dataset

#Define number of hidden layers and number of neurone units in each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60
x = tensorflow.placeholder(tensorflow.float32 , [None , n_dim])
w = tensorflow.Variable(tensorflow.zeros([n_dim , n_class]))
b = tensorflow.Variable(tensorflow.zeros([n_class]))
y = tensorflow.placeholder(tensorflow.float32 , [None , n_class])

#Build Model
#Build weights and biases for each layer
#Initialise variable nodes
init = tensorflow.global_variables_initializer()

#Build loss function
#Build Session
