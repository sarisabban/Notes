#!/usr/bin/python3

import keras
from keras.preprocessing.image import ImageDataGenerator

'''
train_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/train' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 10)
tests_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/tests' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 10)
valid_image = ImageDataGenerator().flow_from_directory('../OLD/CatDog/valid' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 4)
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
#model.fit_generator(x_train , y_train , batch_size = 128 , epochs = 12 , verbose = 1 , validation_data (x_test , y_test))

'''
Standard CNN Models:

Resnet50
keras.applications.resnet50

VGG19
keras.applications.vgg19
'''
