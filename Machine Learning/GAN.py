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




#Generate Image
r, c = 5, 5									# 
noise = numpy.random.normal(0, 1, (r * c, 100))					# 
gen_imgs = G.predict(noise)							# Generate a image (predition) from the neural network
gen_imgs = 0.5 * gen_imgs + 0.5							# Rescale images 0 - 1
fig, axs = matplotlib.pyplot.subplots(r, c)					# 
cnt = 0										# 
for i in range(r):								# 
	for j in range(c):							# 
		axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')		# 
		axs[i,j].axis('off')						# 
		cnt += 1							# 
fig.savefig('generated.png')							# Save image
matplotlib.pyplot.close()							# Close plot
from IPython.display import Image ; Image("generated.png")			# Show Image in Jupyter Notebook
