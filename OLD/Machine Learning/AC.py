# pip install numpy tensorflow gym gym[atari,accept-rom-license]==0.21.0

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('CartPole-v0')      # Features = [0, 0, 0, 0] | Actions = Discrete(2)
env.seed(42)

Y            = 0.99                # The discount factor
max_n_steps  = 10000               # Max n trajectory steps per episode
eps          = 1e-7                # Smallest number such that 1.0 + eps != 1.0
epochs       = 500                 # Number of epochs
features     = 4                   # Number of features
actions      = env.action_space.n  # Number of actions

# Buffer
A_history    = []                  # List history of actions
Q_history    = []                  # List history of q-values
R_history    = []                  # List history of reward
Reward       = 0                   # Reward count
episode      = 0                   # Episode count

# Actor neural network (for playing)
actor = keras.Sequential()
actor.add(keras.layers.Input(shape=(features,)))
actor.add(keras.layers.Dense(128, activation='relu'))
actor.add(keras.layers.Dense(actions, activation='softmax'))

# Critic neural network (for q-value)
critic = keras.Sequential()
critic.add(keras.layers.Input(shape=(features,)))
critic.add(keras.layers.Dense(128, activation='relu'))
critic.add(keras.layers.Dense(1, activation='softmax'))

# Choose the optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function = keras.losses.Huber()

# Train
for episode in range(epochs):
	S = np.array(env.reset())                                                    # Start new episode get state
	Gt = 0

	with tf.GradientTape(persistent=True) as tape:
		for ntimestep in range(1, max_n_steps):                                  # Loop to get a trajectory
			state_tensor = tf.convert_to_tensor(S)                               # Convert S numpy tensor into a tensorflow tensor
			state_tensor = tf.expand_dims(state_tensor, 0)                       # Expland by 1 dimention

			# Take action
			action_probs = actor(state_tensor)                                   # Get action probabilities from actor network 
			q_value = critic(state_tensor)                                       # Get q-value from critic network
			Q_history.append(q_value[0, 0])                                      # Save q-value 
			action = np.random.choice(actions, p=np.squeeze(action_probs))		 # Choose action randomly
			A_history.append(tf.math.log(action_probs[0, action]))               # Save chosen action
			S2, R, St, _ = env.step(action)                                      # Apply action
			R_history.append(R)                                                  # Save reward
			Gt += R                                                              # Update episode reward

			if St:                                                               # End trajectory if St becomes True
				break
		
		# Calculate timestep rewards
		returns = []                                                             # List of discounted rewards -> at each n time step: this is the discounted rewards at that point, these values are the labels for the critic
		discounted_sum = 0
		for r in R_history[::-1]:                                                # Loop backwards because we are discounting each reward in an episode
			discounted_sum = r + Y * discounted_sum                              # Discount reward
			returns.insert(0, discounted_sum)

		# Normalize discounted rewards
		returns = np.array(returns)
		returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
		returns = returns.tolist()

		# Calculate loss
		# At this point in history, the critic estimated that we would get a
		# Gt = `value` in the future. We took an action with log probability
		# of `log_prob` and ended up recieving a total reward = `ret`.
		# The actor must be updated so that it predicts an action that leads to
		# high rewards (compared to critic's estimate) with high probability.
		actor_losses = []
		critic_losses = []
		for log_prob, value, ret in zip(A_history, Q_history, returns):
			diff = ret - value                                                   # Difference between return and value
			actor_losses.append(-log_prob * diff)                                # The actor losses
			critic_losses.append(loss_function(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))) # The critic losses, critic must be updated so that it predicts a better q-values

		# Backpropagation
		loss_value = sum(actor_losses) + sum(critic_losses)                      # Calculate loss between actor and critic
		grads = tape.gradient(loss_value, actor.trainable_variables)             # Gradient of actor
		optimizer.apply_gradients(zip(grads, actor.trainable_variables))         # train actor
		grads = tape.gradient(loss_value, critic.trainable_variables)            # Gradient of critic
		optimizer.apply_gradients(zip(grads, critic.trainable_variables))        # train critic

		# Clear the loss and reward history
		A_history.clear()
		Q_history.clear()
		R_history.clear()

		# Check if solved
		Reward = 0.05 * Gt + (1 - 0.05) * Reward                                 # Check if we have reached a desirable average Gt
		if Reward > 195:  break                                                  # Condition to consider the task solved

		# Log
		episode += 1
		output = 'running reward: {:.2f} at episode {}'
		print(output.format(Reward, episode))
