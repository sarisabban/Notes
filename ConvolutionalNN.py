import numpy , random , keras , pandas , sklearn , scipy

from keras.preprocessing.image import ImageDataGenerator

#Import data - pip3 install pillow
#Keras simply can take images from directories and organises them according to class, this makes importing the data quick and very simple. The target size is the size of each picture's pixles that we will import, i.e: we can have different picture sizes but we will import only specific pixels so that the dataset has uniform examples.
train_image = ImageDataGenerator().flow_from_directory('/home/acresearch/Desktop/CatDog/train' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 10)
valid_image = ImageDataGenerator().flow_from_directory('/home/acresearch/Desktop/CatDog/valid' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 4)
tests_image = ImageDataGenerator().flow_from_directory('/home/acresearch/Desktop/CatDog/tests' , target_size = (224 , 224) , classes = ['cat' , 'dog'] , batch_size = 10)

#gen = ImageDataGenerator(rotation_range = 10 , width_shift_range = 0.1 , height_shift_range = 0.1 , shear_range = 0.15 , zoom_range = 0.1 , channel_shift_range = 0.15 , horizontal_flip = True)	#Augment the images (flip them, zoom them etc...)
#pix = numpy.expand_dims(scipy.ndimage.imread('28.jpg') , 0)																																			#Change image dimentions to be used in next line
#aug = gen.flow(pix)																																													#Augment this particular image
#img = [next(aug)[0].astype(numpy.uint8) for i in range(10)]																																			#Generate 10 different augmented images of this particular image

#Setup neural network
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32 , (3 , 3) , input_shape = (224 , 224 , 3) , activation = 'relu'))																									#Hidden layer 1 is a 2D convolutional layer (because 2D images). 32 is the output filters in the convolution. (3 , 3) is the kernel size, the dimentions of the convolution window that will move through the picture. The input_shape of the picture highet , width , channel (3 for RGB)
model.add(keras.layers.Flatten())																																									#A flatten layer that will flatten our previous layer into a 1D tensor to be inserted into a dense later
model.add(keras.layers.Dense(2 , activation = 'softmax'))																																			#A dense output layer with 2 nodes (because 2 classes)
#print(model.summary())

#Compile model
model.compile(keras.optimizers.Adam() , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

#Train model
model.fit_generator(train_image , steps_per_epoch = 4 , validation_data = valid_image , validation_steps = 4 , epochs = 5 , verbose = 2)															#fit_generator fits a model on data generated batch by batch (from the data that grabs only a batch of images at a time). steps_per_epoch is total number of batches of samples yeild from the generator before an epoch cycles is declaired complete, because 40 images in total (20 cat 20 dog) in the training set and a batch size of 10 pictures at a time then it will take 4 batches to complete 1 epoch through the whole training dataset. Validation using the validation set. validation_steps is just like the steps_per_epoch but for the validation set. 

#Prediction
prediction = model.predict_generator(tests_image , steps = 1 , verbose = 0)
print(prediction)





def VGG_16():	#Award winning CNN (but doesn't work here)
#Setup neural network (VGG16)
	model = keras.models.Sequential()
	model.add(keras.layers.ZeroPadding2D((1 , 1),input_shape = (224 , 224 , 3)))
	model.add(keras.layers.Convolution2D(64 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(64 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.MaxPooling2D((2 , 2) , strides = (2 , 2)))

	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(128 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(128 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.MaxPooling2D((2 , 2) , strides = (2 , 2)))

	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(256 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(256 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(256 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.MaxPooling2D((2 , 2) , strides = (2 , 2)))

	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.MaxPooling2D((2 , 2) , strides = (2 , 2)))

	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.ZeroPadding2D((1 , 1)))
	model.add(keras.layers.Convolution2D(512 , 3 , 3 , activation = 'relu'))
	model.add(keras.layers.MaxPooling2D((2 , 2) , strides = (2 , 2)))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(4096 , activation = 'relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(4096 , activation = 'relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(2 , activation = 'softmax'))
