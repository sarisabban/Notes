#!/usr/bin/python3
#https://github.com/eriklindernoren/Keras-GAN#gan
#https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

import matplotlib.pyplot
import numpy
import keras

#Import Data
from keras.datasets import mnist
(X, _), (_, _) = mnist.load_data()						# Import data
X = X / 127.5 - 1.								# Make all tensor values between -1 and 1
X = numpy.expand_dims(X, axis=3)						# Add an extra dimention for a channel
img_shape = (28, 28, 1)								# Identify shape of data
latent_dim = 100								# Size of noise
batch_size = 32
epochs = 30000

#Discriminator
D = keras.models.Sequential()
D.add(keras.layers.Flatten(input_shape=img_shape))				# For am image flatten the tensor into a vector
D.add(keras.layers.Dense(512))
D.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
D.add(keras.layers.Dense(256))
D.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
D.add(keras.layers.Dense(1, activation='sigmoid'))				# Output is either True or False
D.summary()

#Generator
G = keras.models.Sequential()
G.add(keras.layers.Dense(256, input_dim=latent_dim))				# Input noise
G.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
G.add(keras.layers.BatchNormalization(momentum=0.8))
G.add(keras.layers.Dense(512))
G.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
G.add(keras.layers.BatchNormalization(momentum=0.8))
G.add(keras.layers.Dense(1024))
G.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
G.add(keras.layers.BatchNormalization(momentum=0.8))
G.add(keras.layers.Dense(numpy.prod(img_shape), activation='tanh'))		# Output nodes for each data item in data vector
G.add(keras.layers.Reshape(img_shape))						# Reshape output vector to shape of image
G.summary()

#Discriminator Model
DM = keras.models.Sequential()
DM.add(D)
DM.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])

#Adversarial Model
AM = keras.models.Sequential()
AM.add(G)
AM.add(D)
AM.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8), metrics=['accuracy'])

#Training
for epoch in range(epochs):
	#Generate a fake image
	images_train = X[numpy.random.randint(0, X.shape[0], size=batch_size)]	# Generate a real image tensor with shape (batch size, row, column, channel)
	noise = numpy.random.uniform(-1.0, 1.0, size=[batch_size, 100])		# Generate a noise tensor with shape (Batch size, length of noise vector[100]) and fill it with random numbers between -1 and 1
	images_fake = G.predict(noise)						# Use noise tensor to generate a prediction (batch size of fake images) from the Generator neural network, the output shape of course will be (batch size, row, columns, channel)
	#Train discriminator
	x = numpy.concatenate((images_train, images_fake))			# Combine the real image tensor (from the dataset) with the fake images tensor (from the generator) to generate a dataset with real images and fake images
	y = numpy.ones([2*batch_size, 1])					# Generate the labels tensor with twice the size of the batch (because we combined two batches real and fake) and give them all the label of 1 (True)
	y[batch_size:, :] = 0							# For real images keep 1 (True) but for fake images replace 1 with 0 (False)
	d_loss = DM.train_on_batch(x, y)					# Feed the training (real+fake) and label sets into the Discriminator neural network to train it to find the difference between fake and real data
	#Train adversarial
	y = numpy.ones([batch_size, 1])						# After the Discriminator is trained, now generate a new labels tensor and give it all the label of 1 (True)
	a_loss = AM.train_on_batch(noise, y)					# Train Adversarial neural network, can the discriminator find the fake image from the noise or not? if yes update weights of generator to become better at faking images
	D_loss = round(float(d_loss[0]), 3)
	D_accu = round(float(d_loss[1]), 3)
	A_loss = round(float(a_loss[0]), 3)
	print ('{} [D loss: {}, accuracy: {}] [G loss: {}]'.format(epoch, D_loss, D_accu, A_loss))

#Save Model
G.save_weights('GAN.h5')
#Load model and weights
#G.load_weights('GAN.h5')

#Generate 25 Images
r, c = 5, 5									# Rows, columns number of images (5 images by 5 images = 25 images)
noise = numpy.random.normal(0, 1, (r * c, 100))					# Generate a noise matrix, with a shape of (number of images , number of noise items per image), and values between -1 and 1, because loc = 0 (center of distribution) and scale = 1 (standard deviation, i.e: +1 and -1 from 0), the size will be 25 images (5*5) each with 100 items of noise
gen_imgs = G.predict(noise)							# Generate an image (predition) from the neural network using the noise matrix as input (number of images , rows of each image , columns of each image , channels)
gen_imgs = 0.5 * gen_imgs + 0.5							# Rescale images from -1.0-1.0 to 0.0-1.0
fig, axs = matplotlib.pyplot.subplots(r, c)					# Setup a subplot [25 plots (images) within a plot]
cnt = 0										# Count images
for i in range(r):								# For number of row images
	for j in range(c):							# For number of column images
		axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')		# Plot tensors
		axs[i,j].axis('off')						# Remove plot axis
		cnt += 1							# Add count for image number
matplotlib.pyplot.show()							# Show the plot with the images
#fig.savefig('generated.png')							# Save plot as a .png image file
matplotlib.pyplot.close()							# Close plot
#from IPython.display import Image ; Image("generated.png")			# Show Image in Jupyter Notebook
'''
#Generate an Image
noise = numpy.random.normal(0, 1, (1, 100))					# Generate a noise matrix, with a shape of (number of images , number of noise items per image), and values between -1 and 1, because loc = 0 (center of distribution) and scale = 1 (standard deviation, i.e: +1 and -1 from 0), the size will be 1 image with 100 items of noise
gen_imgs = G.predict(noise)							# Generate an image (predition) from the neural network using the noise matrix as input (number of images , rows of each image , columns of each image , channels)
gen_imgs = 0.5 * gen_imgs + 0.5							# Rescale images from -1.0-1.0 to 0.0-1.0
gen_imgs = gen_imgs.reshape([28, 28])						# remove batch number and channels (only keep rows and columns)
matplotlib.pyplot.imshow(gen_imgs, cmap='gray')					# Plot tensor with gray scale rather than colours
matplotlib.pyplot.axis('off')							# Remove plot axis 
matplotlib.pyplot.show()							# Show the plot with the image
#matplotlib.pyplot.savefig('generated.png')					# Save plot as a .png image file
'''

''' # For RamaNet
real = X[np.random.randint(0, X.shape[0], size=batchs)]					# Real data from dataset, taking random sections from the dataset, shape is (batch size - number of examples, amino acid size, channels)
noise = np.random.uniform(0.0, 1.0, size=[batchs, 100])					# Noise data between 0 and 1 (because dataset is normalised at 0-1), shape is (batch size, noise size)
fake = G.predict(noise)													# Take noise and generate a tensor that looks like real data, shape is same as real data (batch size, amino acids size, channels)
# Now I have real data and fake data
x = np.concatenate((real, fake))										# Combine both real and fake data into 1 tensor, thus a dataset that contains both real and fake data
y = np.ones([2*batchs, 1])												# Label all real data with 1 (true)
y[batchs:, :] = 0														# Label all fake data with 0 (false)
# Now I have a labeled real and fake data
d_loss = DM.train_on_batch(x, y)										# Train Discriminator to learn what is real and what is fake data
# After training the D now it is time to update the G
y = np.ones([batchs, 1])												# Make a new label of all true, indipendent of previous y label
a_loss = AM.train_on_batch(noise, y)									# Train both D and G: use the generated noise with True label and 
'''
