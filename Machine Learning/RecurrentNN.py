#!/usr/bin/python3
#https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
import sys , numpy , random , keras

## Char2Vec vectorising charachter to predict the next charachter in a sentance, but we can vectorise words in Word2Vec to predicts the next word in a sentace using the same concept but slightly different script ##
#Import text
text = open('../OLD/nietzsche.txt').read().lower()
#Process the text to be integer encoded
chars = sorted(list(set(text)))					#make a list of all the different charachters in the entire text
chars.insert(0 , '\0')						#sometimes it is useful to have a zero value as a charachter, used as padding when needed
vocab_size = len(chars) + 1					#number of unique charachters in the text
chars_indices = dict((c , i) for i , c in enumerate(chars))	#give every charachter a unique integer ID
indices_chars = dict((i , c) for i , c in enumerate(chars))	#give every unique integer ID a charachter (opposite of previous line and used to translate back to words)
dataset = [chars_indices[c] for c in text]			#process all text charachers to be integer encoded (not used here)
#Cut the text into overlapping sequences
maxlen = 40							#Max charachter length of a sequence
step = 3							#Each sequence moves by 3 charachters relative to previous sequence
sentences = list()
next_chars = list()
for i in range(0 , len(text) - maxlen , step):			#Length of text - 40 and move 3 values each loop
	sentences.append(text[i : i + maxlen])			#Slice this range into an item and append it to the sentences list
	next_chars.append(text[i + maxlen])			#Take the "next charachter" that comes after the sentace in the previous list and append it into a new list. That way we have a list of sentances and another list of charachters that come after each sentance
#Vectorise - The dataset now has the shape (number of sentances , Maximum sentance length , number of available charachters)
X = numpy.zeros((len(sentences) , maxlen , len(chars)) , dtype = numpy.bool)
Y = numpy.zeros((len(sentences) , len(chars)) , dtype = numpy.bool)
#One-hot encoding - True for the charachter that comes after the sequence
for i , sentence in enumerate(sentences):
	for t , char in enumerate(sentence):
		X[i , t , chars_indices[char]] = 1
	Y[i , chars_indices[next_chars[i]]] = 1

#TensorBoard log
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars))))
model.add(keras.layers.Dense(len(chars) , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model - Accuracy is not that important in NLP because it is relative. What is more important is the language output
model.fit(X , Y , batch_size = 128 , verbose = 2 , epochs = 1 , callbacks = [tensorboard])

#Save Model
model.save('model.h5')

#Load Model
#model.load_weights('model.h5')

#Generate text from the trained model- Start by randomly generate a starting sentance
print('--------------------')
start_index = random.randint(0 , len(text) - maxlen - 1)	#Choose a random number from 0 to length of text - max charachter length - 1. This gives a number that is that start of a poisiton in the text
sentence = text[start_index : start_index + maxlen]		#Use that position to slice out a sentance from start_index position with a length of maxlen max charachter length. This gives a random sentance from the text
print(sentence)							#Print the stating sentance
for iter in range(400):						#Move 400 steps. Controls length of text
	#One-hot encode that sentance
	x_pred = numpy.zeros((1 , maxlen , len(chars)))		#Generate a tensor that has the same of (1 , max sentance length , number of available charachter), in our example: (1 , 40 , 58). The tensor is just filled with zeros, no other information
	for t , char in enumerate(sentence):			#Loop through the randomly generated sentance
		x_pred[0 , t , chars_indices[char]] = 1.0	#One-hot encode the randomly generated sentance (put a 1.0 for each charachter as available from the list of charachters)
	#Use that tensor to make a prediction of the next charachter that comes after the randomly generated sentance
	preds = model.predict(x_pred , verbose = 0)[0]		#Returns a vector of shape (number of available charachter,) with the values of the probability of each charachter being the next charachter after the randomly generated sentance
	#Decode that character
	temperature = 0.2					#Temperature is used to make lower probabilities lower and higher probabilities higher (using Temperature < 1.0) or vise versa (using Temperature < 1.0). Calibrate until the best temperatures is achieved. Temperature of 1 does nothing
	preds = numpy.asarray(preds).astype('float64')		#Make sure all tensor values are float64
	preds = numpy.log(preds) / temperature			#Log each tensor value and then divide each value by the temperature
	exp_preds = numpy.exp(preds)				#Turn each tensor value into an exponant
	preds = exp_preds / numpy.sum(exp_preds)		#Re-Normalise (all values add up to 1) by dividing the exponant values by the sum of all the values
	probas = numpy.random.multinomial(1 , preds , 1)	#Randomly choose one index based on probability (most times it will choose the index with the highest probability, but sometimes it will randomly choose a slightly lower one)
	next_index = numpy.argmax(probas)			#Choose the largest value's number location in the vector, which will correspond to the identify of the charachter from the charachter list "indices_chars"
	next_char = indices_chars[next_index]			#Find the value's corresponding charachter
	sentence = sentence[1 : ] + next_char			#Add new charachter to sentance and remove 1 charachter from start of the sentence to maintain its length
	sys.stdout.write(next_char)				#Print the generated charachters from the neural network prediction
	sys.stdout.flush()					#Flush terminal buffer, this and the previous line allows for the charachters to be printer like a type writer (one at a time)
print('\n--------------------')



'''
Stateful: Natural Language Processing
* Remembers information between batches
* It memorises, it keeps the internal gates' states as they are between batches
* Used in datasets that has all its information related to each other, in other words when two sequences in two different batches have connections, like stocks (previous price of stock affects next price of stock)

Stateless: Stocks
* Does not remember information between batches
* It does not memorise, it resets all internet gates between batches
* Used in datasets that has all its information not related to each other, in other words when two sequences in two different batches do not have connections, like sentances in language (previous sentance does not affects next sentance, they are indipendant of each other)

In both types the final hidden layers' nodes' weights are kept. It is just a matter of whether or not the internal gates' states are kept or reset

Code:
model.add(keras.layers.LSTM(128 , stateful = True))
'''

'''
Batch Size:
Strongly impacts the prediction accuracy, 60-80 is usually optimal (try to have multiples of 8).
Training set size must be divisible by batch size without remainder because batches are smaller training units. If training set is size X what will be the batch size to become 10% of that training set but still divisile without a remainder?
Low = longer
High= faster
for sentances that are not equal in length (for example protein sequences as sentences) we can use a batch size = 1 instead of padding the sentances to make them all the same length

Time Step:
value of time steps in the past used to predict the same value of time in the future. And this moving window slides by only 1 time step in the future through the dataset (called striding). Just like NLP with sentences. Used with stateful LSTMs
for i in range(timestep , len(text) + timestep):
	X_train.append(training_set_scaled[i - timestep : i , 0])
	Y_train.append(training_set_scaled[i : i + timestep , 0])
Gives the shape of (number of example , timestep , parameters to predict)

Epochs:
Low = not achieve enough accuracy
High= overfitting
'''

'''
Embedding:
A word embedding is a class of approaches for representing words and documents using a dense vector representation
					input_dim	output_dim		input_length
model.add(keras.layers.Embedding(size of vocabulary , number of sentances (also known as a vector space in which words will be embedded) , input_length = words in each sentance))
Requires that is input text data have the words to be integer encoded (just lines 8 - 14)
Two popular examples of methods of learning word embeddings from text:
* Word2Vec
* GloVe
input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words
output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem
input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000
If you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer

https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
Let us pretend that we have 10 documents, and each documents is full of words (but here each document only has 2 words). Each document is classified as positive “1” or negative “0”
---------
#!/usr/bin/python3
import sys , numpy , random , keras
from keras.preprocessing import text

# Define documents
docs = ['Well done!',
	'Good work',
	'Great effort',
	'nice work',
	'Excellent!',
	'Weak',
	'Poor effort!',
	'not good',
	'poor work',
	'Could have done better.']

# Define class labels
labels = numpy.array([1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0])

# Integer encode each word in each document (making all the same words in different documents have the same integer encoding)
vocab_size = 50						# We will estimate the vocabulary size of 50, which is much larger than needed to reduce the probability of collisions from the hash function
encoded_docs = [text.one_hot(d , vocab_size) for d in docs]
print(encoded_docs)

#Padding - The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length, therefore here we use padding of 4 to get all document lengths into 4 maximum words long
max_length = 4
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs , maxlen = max_length , padding = 'post')
print(padded_docs)

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size , 8 , input_length = max_length))					#The Embedding has a vocabulary of 50 (vocab_size) and an input length of 4 (max_length). We will choose a small embedding space of 8 dimensions (8 number of sentances)
model.add(keras.layers.Flatten())										#Flatten to a one dimentional vector of 32 -> 4 x 8 matrix and this is squashed to a 32-element vector
model.add(keras.layers.Dense(1 , activation = 'sigmoid'))							#Output is either 0 or 1
model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
model.fit(padded_docs , labels , epochs = 50, verbose = 2)
---------
'''
