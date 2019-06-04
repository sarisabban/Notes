import os
import keras
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

def GAN_PSC(choice='generate'):
	'''
	A generative adversarial network that generates novel unnatural protein
	topologies. This network uses the phi and psi angles as well as the Ca
	distances as protein structure features
	'''
	lrG			= 0.0002
	lrD			= 0.0002
	lossG		= 'binary_crossentropy'
	lossD		= 'binary_crossentropy'
	decayG		= 1e-8
	decayD		= 1e-8
	nodeG		= 8
	nodeD		= 8
	dropoutG	= 0.0
	dropoutD	= 0.0
	alphaG		= 0.2
	alphaD		= 0.2
	momentumG	= 0.8
	momentumD	= 0.8
	latent		= 100
	batchs		= 32
	epochs		= 10000
	#-----------------
	(dataset, _), (_, _) = mnist.load_data()
	shape = (28, 28, 1)
	dataset = dataset / 127.5 - 1.
	dataset = np.expand_dims(dataset, axis=3)
	G = keras.models.Sequential()
	G.add(keras.layers.Dense(2**(nodeD+0), input_dim=latent))
	G.add(keras.layers.LeakyReLU(alpha=alphaG))
	G.add(keras.layers.BatchNormalization(momentum=momentumG))
	G.add(keras.layers.Dense(2**(nodeD+1)))
	G.add(keras.layers.LeakyReLU(alpha=alphaG))
	G.add(keras.layers.BatchNormalization(momentum=momentumG))
	G.add(keras.layers.Dense(2**(nodeD+2)))
	G.add(keras.layers.LeakyReLU(alpha=alphaG))
	G.add(keras.layers.BatchNormalization(momentum=momentumG))
	G.add(keras.layers.Dense(np.prod(shape), activation='tanh'))
	G.add(keras.layers.Reshape(shape))
	D = keras.models.Sequential()
	D.add(keras.layers.Flatten(input_shape=shape))
	D.add(keras.layers.Dense(2**(nodeD+1)))
	D.add(keras.layers.LeakyReLU(alpha=alphaD))
	D.add(keras.layers.Dense(2**(nodeD+0)))
	D.add(keras.layers.LeakyReLU(alpha=alphaD))
	D.add(keras.layers.Dense(1, activation='sigmoid'))
	D.compile(optimizer=keras.optimizers.Adam(lr=lrD, decay=decayD), loss=lossD, metrics=['accuracy'])
	z = keras.layers.Input(shape=(latent,))											# Layer of random noise input
	gen = G(z)																		# Layer of generated data input from noise
	D.trainable = False																# Do not train the Discriminator
	validity = D(gen)																# Analyse generated data using discriminator and output probability True or False
	AM = keras.models.Model(z, validity)
	AM.compile(optimizer=keras.optimizers.Adam(lr=lrG, decay=decayG), loss=lossG, metrics=['accuracy'])
	if choice == 'train':
		print('--------------------------------------------------------')
		Epc = []
		DTy = []
		DFy = []
		GNy = []
		y_true = np.ones([batchs, 1])
		y_false = np.zeros([batchs, 1])
		for epoch in range(epochs):
			# Train the Discriminator
			X_real = dataset[np.random.randint(0, dataset.shape[0],size=batchs)]
			X_noise = np.random.normal(0.0, 1.0, size=[batchs, latent])
			X_fake = G.predict(X_noise)
			dT_loss = D.train_on_batch(X_real, y_true)
			dF_loss = D.train_on_batch(X_fake, y_false)
			# Train the Generator
			g_loss = AM.train_on_batch(X_noise, y_true)
			# Prints
			DT_loss = round(float(dT_loss[0]), 3)
			DF_loss = round(float(dF_loss[0]), 3)
			GN_loss = round(float(g_loss[0]), 3)
			Verb =	'Epoch: {:6d} [DT {:.7f}] [DF {:.7f}] [G {:.7f}]'\
					.format(epoch+1, DT_loss, DF_loss, GN_loss)
			#print(Verb)
			Epc.append(epoch)
			DTy.append(DT_loss)
			DFy.append(DF_loss)
			GNy.append(GN_loss)
		G.save_weights('weights.h5')
		return(Epc, DTy, DFy, GNy)
		print('--------------------------------------------------------\nDone')
	if choice == 'generate':
		try: G.load_weights('weights.h5')
		except: print('Missing file: weights.h5'); exit()
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, latent))
		gen_imgs = G.predict(noise)
		gen_imgs = 0.5 * gen_imgs + 0.5
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		plt.show()
		plt.close()

# Train the neural network
Epc, DTy, DFy, GNy = GAN_PSC('train')
fig1 = plt.figure(1)
rect = fig1.patch
rect.set_facecolor('grey')
plt.subplot(1, 1, 1, facecolor='lightslategray')
plt.title(label='Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.plot(Epc, DTy, linewidth=0.75, color='Blue', label='Discriminator True')
plt.plot(Epc, DFy, linewidth=0.75, color='Green', label='Discriminator False')
plt.plot(Epc, GNy, linewidth=0.75, color='Red', label='Generator')
plt.legend(loc='upper right')
plt.show()
plt.close()

# Generate Data
GAN_PSC('generate')
