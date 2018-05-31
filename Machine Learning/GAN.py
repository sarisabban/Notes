#!/usr/bin/python3
#https://github.com/eriklindernoren/Keras-GAN#gan

import keras
import matplotlib.pyplot
import numpy

from keras.datasets import mnist

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

optimizer = keras.optimizers.Adam(0.0002, 0.5)

# Build and compile the discriminator
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=img_shape))
model.add(keras.layers.Dense(512))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(256))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
img = Input(shape=img_shape)
validity = model(img)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# Build the generator
model = keras.models.Sequential()
model.add(keras.layers.Dense(256, input_dim=latent_dim))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
model.add(keras.layers.BatchNormalization(momentum=0.8))
model.add(keras.layers.Dense(512))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
model.add(keras.layers.BatchNormalization(momentum=0.8))
model.add(keras.layers.Dense(1024))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))
model.add(keras.layers.BatchNormalization(momentum=0.8))
model.add(keras.layers.Dense(numpy.prod(img_shape), activation='tanh'))
model.add(keras.layers.Reshape(img_shape))
model.summary()
noise = Input(shape=(latent_dim,))
img = model(noise)

# The generator takes noise as input and generates imgs
z = Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = keras.models.Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

#Training
epochs = 3
batch_size = 32
sample_interval = 200
# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = numpy.expand_dims(X_train, axis=3)

# Adversarial ground truths
valid = numpy.ones((batch_size, 1))
fake = numpy.zeros((batch_size, 1))

for epoch in range(epochs):

	# ---------------------
	#  Train Discriminator
	# ---------------------

	# Select a random batch of images
	idx = numpy.random.randint(0, X_train.shape[0], batch_size)
	imgs = X_train[idx]

	noise = numpy.random.normal(0, 1, (batch_size, 100))

	# Generate a batch of new images
	gen_imgs = generator.predict(noise)

	# Train the discriminator
	d_loss_real = discriminator.train_on_batch(imgs, valid)
	d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
	d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

	# ---------------------
	#  Train Generator
	# ---------------------

	noise = numpy.random.normal(0, 1, (batch_size, 100))

	# Train the generator (to have the discriminator label samples as valid)
	g_loss = combined.train_on_batch(noise, valid)

	# Plot the progress
	print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

#Generate Image
r, c = 5, 5
noise = numpy.random.normal(0, 1, (r * c, 100))
gen_imgs = generator.predict(noise)
# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = matplotlib.pyplot.subplots(r, c)
cnt = 0
for i in range(r):
	for j in range(c):
		axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
		axs[i,j].axis('off')
		cnt += 1
fig.savefig("%d.png" % int(epoch))
matplotlib.pyplot.close()
