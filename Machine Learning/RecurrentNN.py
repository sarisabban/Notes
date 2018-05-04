#!/usr/bin/python3
#https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
import sys , numpy , random , keras , io

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
model.compile(keras.optimizers.RMSprop(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model - Accuracy is not that important in NLP because it is relative. What is more important is the language output
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')
#model.fit(X , Y , batch_size = 128 , verbose = 2 , epochs = 60 , callbacks = [tensorboard])


















def sample(preds , temperature = 1.0):
	# helper function to sample an index from a probability array
	preds = numpy.asarray(preds).astype('float64')
	preds = numpy.log(preds) / temperature
	exp_preds = numpy.exp(preds)
	preds = exp_preds / numpy.sum(exp_preds)
	probas = numpy.random.multinomial(1, preds, 1)
	return numpy.argmax(probas)


def on_epoch_end(epoch , logs):
	# Function invoked at end of each epoch. Prints generated text.
	print('----- Generating text after Epoch: %d' % epoch)

	start_index = random.randint(0, len(text) - maxlen - 1)
	for diversity in [0.2, 0.5, 1.0, 1.2]:
		print('----- diversity:', diversity)

		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)

		for i in range(400):
			x_pred = numpy.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
				x_pred[0, t, chars_indices[char]] = 1.

			preds = model.predict(x_pred , verbose = 0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_chars[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

			sys.stdout.write(next_char)
			sys.stdout.flush()




#model.fit(x, y,batch_size=128,epochs=60,callbacks=[print_callback])
