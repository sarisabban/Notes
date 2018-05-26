import pandas
import numpy
import keras

#Import text
data = pandas.read_csv('SS.csv' , sep=';')
column = data['Secondary_Structures']
text = '\n'.join(column)
chars = sorted(list(set(text)))
chars_indices = dict((c , i) for i , c in enumerate(chars))
indices_chars = dict((i , c) for i , c in enumerate(chars))

#Generate sentences and next characters
maxlen = 70
step = 1
sentences = []
next_chars = []
for i in range(0 , len(text) - maxlen , step):
	sent = text[i : i + maxlen]
	char = text[i + 1 : i + maxlen + 1]
	sentences.append(sent)
	next_chars.append(char)
	#print(sent , char)

batch_size = 128
validation_split = 0.2
num_batches_total = (len(text) - 1) // (batch_size * maxlen)
num_batches_val = int(num_batches_total * validation_split)
num_batches_train = num_batches_total - num_batches_val



def make_sentences(text, num_batches):
	sentences = []
	for batch_num in range(num_batches):
		for i in range(batch_size):
			offset = i * num_batches * maxlen + batch_num * maxlen        
			sentence = text[offset:offset + maxlen + 1]
			sentences.append( sentence )
	return sentences



def make_XY(sentences):
	X = numpy.zeros((len(sentences), maxlen, len(chars)) , dtype = numpy.bool)
	Y = numpy.zeros((len(sentences), maxlen, 1), dtype = numpy.int32)

	for i , sentence in enumerate(sentences):
		sentence_in = sentence[:-1]
		sentence_out = sentence[1:]

	for t , char in enumerate(sentence_in):
		X[i, t, chars_indices[char]] = 1

	for t , char in enumerate(sentence_out):
		Y[i, t, 0] = chars_indices[char]
      
	return X, Y



train_size = num_batches_train * batch_size * maxlen
sentences_train = make_sentences(text[:train_size + 1], num_batches_train)
sentences_val = make_sentences(text[train_size:], num_batches_val)

X_train, Y_train = make_XY(sentences_train)
X_val, Y_val = make_XY(sentences_val)

#tensorboard = keras.callbacks.TensorBoard(log_dir = '')

model = keras.models.Sequential()
model.add(keras.layers.LSTM(64 , batch_input_shape = (batch_size, maxlen, len(chars)), return_sequences=True, stateful=True))
model.add(keras.layers.core.Dropout(0.25))
model.add(keras.layers.TimeDistributed( keras.layers.Dense(len(chars)) ))
model.add(keras.layers.Activation('softmax'))

model.compile(keras.optimizers.Adam(lr=0.001) , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=batch_size, shuffle=False, verbose=2)
#model.save('model.h5')

#Load Model
#model.load_weights('SS.h5')
#Generate
print('--------------------')
start_index = random.randint(0 , len(text) - maxlen - 1)
sentence = text[start_index : start_index + maxlen]
print('Starting sequence:' , sentence)
for iter in range(1000):
    x_pred = numpy.zeros((32 , maxlen , len(chars)))
    for t , char in enumerate(sentence):
        x_pred[0 , t , chars_indices[char]] = 1.0
    preds = model.predict(x_pred , verbose = 0)[0]
    preds = preds[-1]
    temperature = 1.0
    preds = numpy.asarray(preds).astype('float64')
    preds[preds == 0.0] = 0.0000001
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1 , preds , 1)
    next_index = numpy.argmax(probas)
    next_char = indices_chars[next_index]
    sentence = sentence[1 : ] + next_char
    sys.stdout.write(next_char)
    sys.stdout.flush()
