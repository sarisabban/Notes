#!/usr/bin/python3

import numpy , keras
from keras.preprocessing.image import ImageDataGenerator
							  #Locate images       Reshape the image down to    Identify channels        Identify classes	    Identify batches
train_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/train' , target_size = (224 , 224) , color_mode = 'rgb' , classes = ['cat' , 'dog'] , batch_size = 10)
tests_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/tests' , target_size = (224 , 224) , color_mode = 'rgb' , classes = ['cat' , 'dog'] , batch_size = 10)
valid_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/valid' , target_size = (224 , 224) , color_mode = 'rgb' , classes = ['cat' , 'dog'] , batch_size = 4)
num_classes = 2

#(224 , 224 , 3) - (rows , columns , colours)
print(train_image.image_shape)
print(tests_image.image_shape)
print(valid_image.image_shape)

#(10 , 224 , 224 , 3) - (examples in batch , rows , columns , colours)
print(train_image.next()[0].shape)
print(tests_image.next()[0].shape)
print(valid_image.next()[0].shape)

input_shape = train_image.image_shape

#TensorBoard log
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32 , (3 , 3) , input_shape = input_shape , activation = 'relu'))
model.add(keras.layers.Conv2D(64 , (3 , 3) , activation = 'relu'))
model.add(keras.layers.MaxPooling2D((2 , 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128 , activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adadelta(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model
#steps_per_epoch: Number of batches in an epoch. It should typically be equal to the number of samples of your dataset divided by the batch size 40/10 = 4
#validation_steps: Same concept. It should typically be equal to the number of samples of your validation dataset divided by the batch size. 10/10 = 1
steps_per_epoch = int(train_image.samples / train_image.next()[0].shape[0])
validation_steps = int(tests_image.samples / tests_image.next()[0].shape[0])
model.fit_generator(train_image , steps_per_epoch = steps_per_epoch , epochs = 10 , validation_data = tests_image , validation_steps = validation_steps , verbose = 1 , callbacks = [tensorboard])

#Validate model
model.predict_generator(valid_image , verbose = 1)

'''
(x_train , y_train) , (x_test , y_test) = keras.datasets.mnist.load_data()
num_classes = 10
img_rows , img_cols = 28 , 28
#Add colour dimensions
if keras.backend.image_data_format() == 'channel_first':
	x_train = x_train.reshape(x_train.shape[0] , 1 , img_rows , img_cols)
	x_test = x_test.reshape(x_test.shape[0] , 1 , img_rows , img_cols)
	input_shape = (1 , img_rows , img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0] , img_rows , img_cols , 1)
	x_test = x_test.reshape(x_test.shape[0] , img_rows , img_cols , 1)
	input_shape = (img_rows , img_cols , 1)
#Normalising - #(10000 , 28 , 28 , 1) - (examples , rows , columns , colours)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 225
x_test /= 225

#One-hot encoding
y_train = keras.utils.to_categorical(y_train , num_classes)
y_test = keras.utils.to_categorical(y_test , num_classes)

#TensorBoard log
tensorboard = keras.callbacks.TensorBoard(log_dir = './logs')

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32 , (3 , 3) , input_shape = input_shape , activation = 'relu'))
model.add(keras.layers.Conv2D(64 , (3 , 3) , activation = 'relu'))
model.add(keras.layers.MaxPooling2D((2 , 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128 , activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes , activation = 'softmax'))

#Compile model
model.compile(keras.optimizers.Adadelta(lr = 0.01) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#Train model
model.fit(x_train , y_train , batch_size = 128 , epochs = 12 , verbose = 1 , callbacks = [tensorboard] , validation_data = (x_test , y_test))
'''

'''
Standard CNN Models:

Resnet50
keras.applications.resnet50

VGG19
keras.applications.vgg19
'''
