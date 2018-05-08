#!/usr/bin/python3
#https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
import sys , numpy , random , keras

#Import text
text = open('../OLD/nietzsche.txt').read().lower()

#Process the text to be integer encoded
chars = sorted(list(set(text)))					#make a list of all the different charachters in the entire text
chars.insert(0 , '\0')						#sometimes it is useful to have a zero value as a charachter, used as padding when needed
vocab_size = len(chars) + 1					#number of unique charachters in the text
chars_indices = dict((c , i) for i , c in enumerate(chars))	#give every charachter a unique integer ID
indices_chars = dict((i , c) for i , c in enumerate(chars))	#give every unique integer ID a charachter (opposite of previous line and used to translate back to words)
dataset = [chars_indices[c] for c in text]			#process all text charachers to be integer encoded

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

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128 , input_shape = (maxlen , len(chars))))
model.add(keras.layers.Dense(len(chars) , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adam(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model - Accuracy is not that important in NLP because it is relative. What is more important is the language output
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')
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
	#Decode that charachter
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

'''
