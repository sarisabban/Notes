import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.set_random_seed(1)

(dataset, _), (_, _) = mnist.load_data()
dataset = dataset / 127.5 - 1.
dataset = np.expand_dims(dataset, axis=3)
shape = dataset.shape[1:]
#-------------------------
batchs		= 5
clipvalue1	= 1.0
clipvalue2	= 1.0
decay1		= 6e-8
decay2		= 3e-8
epochs		= 3
latent  	= 2
lr1			= 8e-4
lr2			= 4e-4
momentum	= 0.8
node1		= 8
node2		= 8
#-------------------------
latent = int(100*latent)
node1 = int(node1)
node2 = int(node2)
batchs = 2**int(batchs)
G = keras.models.Sequential()
G.add(keras.layers.Dense(2**(node1+0), input_dim=latent, activation='relu'))
G.add(keras.layers.BatchNormalization(momentum=momentum))
G.add(keras.layers.Dense(2**(node1+1), activation='relu'))
G.add(keras.layers.BatchNormalization(momentum=momentum))
G.add(keras.layers.Dense(2**(node1+2), activation='relu'))
G.add(keras.layers.BatchNormalization(momentum=momentum))
G.add(keras.layers.Dense(np.prod(shape), activation='tanh'))
G.add(keras.layers.Reshape(shape))
D = keras.models.Sequential()
D.add(keras.layers.Flatten(input_shape=shape))
D.add(keras.layers.Dense(2**(node2+1), activation='relu'))
D.add(keras.layers.Dense(2**(node2+0), activation='relu'))
D.add(keras.layers.Dense(1, activation='sigmoid'))
DM = keras.models.Sequential()
DM.add(D)
AM = keras.models.Sequential()
AM.add(G)
AM.add(D)
DM.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=lr1, clipvalue=clipvalue1, decay=decay1), metrics=['accuracy'])
AM.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=lr2, clipvalue=clipvalue2, decay=decay2), metrics=['accuracy'])

for epoch in range(1000*int(epochs)):
	images_train = dataset[np.random.randint(0, dataset.shape[0], size=batchs)]
	noise = np.random.uniform(-1.0, 1.0, size=[batchs, latent])
	images_fake = G.predict(noise)
	x = np.concatenate((images_train, images_fake))
	y = np.ones([2*batchs, 1])
	y[batchs:, :] = 0
	d_loss = DM.train_on_batch(x, y)
	y = np.ones([batchs, 1])
	a_loss = AM.train_on_batch(noise, y)
	D_loss = round(float(d_loss[0]), 3)
	D_accu = round(float(d_loss[1]), 3)
	A_loss = round(float(a_loss[0]), 3)
	A_accu = round(float(a_loss[1]), 3)
	print (	'Epoch:{:6d} [Dis Loss:{:.5f}] [Adv Loss:{:.5f}]'\
			.format(epoch+1, D_loss, D_accu, A_loss, A_accu))

#G.save_weights('GAN.h5')
#G.load_weights('GAN.h5')
r, c = 5, 5
noise = np.random.normal(0, 1, (r * c, 100))
gen_imgs = G.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = matplotlib.pyplot.subplots(r, c)
cnt = 0
for i in range(r):
	for j in range(c):
		axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
		axs[i,j].axis('off')
		cnt += 1
matplotlib.pyplot.show()
matplotlib.pyplot.close()
