#!/usr/bin/python3

import numpy , random , keras , pandas , sklearn

#Import text
text = open('/home/acresearch/Desktop/Notes/OLD/nietzsche.txt').read()

#Process the text





'''
#Import data - For language processing the dataset needs to be integer encoded using th keras Tokenizer API (not written here yet)
Max_Features = 20000			#20,000 most common items in the vocabulary
MaxLen = 80				#Maximum length of 80 for features (if less add 0s, if longer crop them)
(x_train , y_train) , (x_test , y_test) = keras.datasets.imdb.load_data(num_words = Max_Features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train , maxlen = MaxLen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test , maxlen = MaxLen)

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.Embedding(Max_Features , 128))						#The Embedding layer can only be used as the first layer and it represents words using a dense vector representation. It requires that the input data be integer encoded. 128 is the number of nodes
model.add(keras.layers.LSTM(128 , dropout = 0.2 , recurrent_dropout = 0.2))
model.add(keras.layers.Dense(1 , activation = 'sigmoid'))

#Compile model
model.compile(keras.optimizers.SGD() , loss = 'binary_crossentropy' , metrics = ['accuracy'])	#Use binary_crossentropy loss because

#Train model
model.fit(x_train , y_train , batch_size = 32 , epochs = 1 , verbose = 2 , validation_data  = (x_test , y_test))
'''
