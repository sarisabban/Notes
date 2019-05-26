import os
import keras
import sklearn
import functools
import bayes_opt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.set_random_seed(1)

(dataset, _), (_, _) = mnist.load_data()
dataset = dataset / 127.5 - 1.
dataset = np.expand_dims(dataset, axis=3)
shape = dataset.shape[1:]
X_train, X_test = train_test_split(dataset, random_state=42)

def create_model(	lr1			= 1e-6,
					lr2			= 1e-6,
					decay1		= 1e-3,
					decay2		= 1e-3,
					clipvalue1	= 1.0,
					clipvalue2	= 1.0,
					latent  	= 1,
					node1		= 6,
					node2		= 7,
					momentum	= 0.1,
					batchs		= 2):
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
	DM.compile(optimizer=keras.optimizers.RMSprop(lr=lr1, clipvalue=clipvalue1, decay=decay1), loss='binary_crossentropy', metrics=['accuracy'])
	AM.compile(optimizer=keras.optimizers.RMSprop(lr=lr2, clipvalue=clipvalue2, decay=decay2), loss='binary_crossentropy', metrics=['accuracy'])
	return(G, DM, AM)

def fit_with(	lr1,
				lr2,
				decay1,
				decay2,
				clipvalue1,
				clipvalue2,
				latent,
				node1,
				node2,
				momentum,
				batchs,
				epochs):
	batchs = 2**int(batchs)
	epochs = 1000*int(epochs)
	G, DM, AM = create_model(	lr1,
								lr2,
								decay1,
								decay2,
								clipvalue1,
								clipvalue2,
								latent,
								node1,
								node2,
								momentum,
								batchs)
	latent = int(100*latent)
	for epoch in range(epochs):
		real = dataset[np.random.randint(0, dataset.shape[0], size=batchs)]
		noise = np.random.normal(0.0, 1.0, size=[batchs, latent])
		fake = G.predict(noise)
		x = np.concatenate((real, fake))
		y = np.ones([2*batchs, 1])
		y[batchs:, :] = 0
		d_loss = DM.train_on_batch(x, y)
		y = np.ones([batchs, 1])
		a_loss = AM.train_on_batch(noise, y)
		D_loss = float(d_loss[0])
		D_accu = float(d_loss[1])
		A_loss = float(a_loss[0])
		A_accu = float(a_loss[1])
		Verb =	'Epoch:{:6d} [Dis Loss:{:.7f}] [Gen Loss:{:.7f}]'\
				.format(epoch+1, D_loss, D_accu, A_loss, A_accu)
		print(Verb)
	return(A_accu)

fit_with_partial = functools.partial(fit_with)
params = {	'lr1'		: (1e-4, 1e-5),	# 8e-4
			'lr2'		: (1e-4, 1e-5),	# 4e-4
			'decay1'	: (1e-8, 1e-9),	# 6e-8
			'decay2'	: (1e-8, 1e-9),	# 3e-8
			'clipvalue1': (0.0, 1.0),	# 1.0
			'clipvalue2': (0.0, 1.0),	# 1.0
			'latent'	: (0.5, 5.0),	# 1
			'node1'		: (1, 10),		# 8
			'node2'		: (1, 10),		# 8
			'momentum'	: (0.0, 0.9),	# 0.8
			'batchs'	: (1, 10),		# 5
			'epochs'	: (3, 6)}		# 3

optimizer = bayes_opt.BayesianOptimization(	f=fit_with_partial,
											pbounds=params,
											verbose=2,
											random_state=1)
optimizer.maximize(init_points=10, n_iter=10)
output = open('Search_Result.txt', 'a')
output.write('| Iteration |  Target  | Params\n')
output.write('|-----------|----------|-----------------------------')
output.write('---------------------------\n')
for i, res in enumerate(optimizer.res):
	line='|{:11}|{:10}| {}\n'.format(i+1, round(res['target'],5), res['params'])
	output.write(line)
output.close()
