# pip install numpy tensorflow gym gym[atari,accept-rom-license]==0.21.0

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('Breakout-v0')  # Features = image (210, 160, 3) | Actions = Discrete(4)
env.seed(42)

Y                      = 0.99                # The discount factor
E                      = 1.0                 # Epsilon value for greedy policy
E_min                  = 0.1                 # End of epsilon of greedy policy
E_interval             = 0.9                 # Interval movement of epsilon
batch                  = 32                  # Batch size from replay buffer
epochs                 = 5                   # Number of epochs
max_n_steps            = 10000               # Max n trajectory steps per episode
actions                = env.action_space.n  # Number of actions

# Replay Buffer
A_history              = []                  # List history of actions
S_history              = []                  # List history of current state
S2_history             = []                  # List history of next state
R_history              = []                  # List history of reward
St_history             = []                  # List history of terminal state
Gt_history             = []                  # List history of episode reward
N                      = 100000              # Max memory length of replay buffer

# Other parameters
Reward                 = 0                   # Reward count
episode                = 0                   # Episode count
frame                  = 0                   # Frame count
random_frames          = 50000               # Number of frames to take random action and observe output
greedy_frames          = 1000000             # Number of frames for exploration
start_train            = 4                   # Start training after 4 action steps
update_QT              = 10000               # Every n steps update target network

# Prediction neural network (for playing - online)
QP = keras.Sequential()
QP.add(keras.layers.Input(shape=(210, 160, 3,)))
QP.add(keras.layers.Conv2D(32, 8, strides=4, activation='relu'))
QP.add(keras.layers.Conv2D(64, 4, strides=2, activation='relu'))
QP.add(keras.layers.Conv2D(64, 3, strides=1, activation='relu'))
QP.add(keras.layers.Flatten())
QP.add(keras.layers.Dense(512, activation='relu'))
QP.add(keras.layers.Dense(actions, activation='linear'))

# Target neural network (for exploring - offline)
QT = QP

# Choose the optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

# Train
for episode in range(epochs):
	S = np.array(env.reset())                                                    # Start new episode get state
	Gt = 0

	for ntimestep in range(1, max_n_steps):                                      # Loop to get a trajectory
		frame += 1                                                               # Count number of frames
		if frame < random_frames or E > np.random.rand(1)[0]:                    # Use E-greedy policy for exploration. If frame count is within the random_frames quantity, or we randomly choose to overcome epsilon. Then randomly take any action with equal probability to explore observation
			action = np.random.choice(actions)                                   # Choose random action
		else:                                                                    # Otherwise do not explore and take action according to the QT exploration network
			state_tensor = tf.convert_to_tensor(S)                               # Convert S numpy tensor into a tensorflow tensor
			state_tensor = tf.expand_dims(state_tensor, 0)                       # Expland by 1 dimention
			action_probs = QP(state_tensor, training=False)                      # Use prediction network to get q-value for each action
			action = tf.argmax(action_probs[0]).numpy()                          # Choose max q-value

		# Take action
		S2, R, St, _ = env.step(action)                                          # Apply action
		state_next = np.array(S2)                                                # Convert next state into a numpy array
		Gt += R                                                                  # Update episode reward

		# Populate replay buffer
		A_history.append(action)                                                 # Save action
		S_history.append(S)                                                      # Save state
		S2_history.append(S2)                                                    # Save next state
		St_history.append(St)                                                    # Save terminal state flag
		R_history.append(R)                                                      # Save reward

		# Episode decays
		S = S2                                                                   # Make S2 as new S
		E -= E_interval / greedy_frames                                          # Decay epsilon
		E = max(E, E_min)                                                        # Choose E as long as it is greater than E_min otherwise choose e_min without decay (bottom)

		# Network training statement
		if frame % start_train == 0 and len(St_history) > batch:                 # Every 4 frames, and after replay buffer contains at lease 1 batch size then start training the neural networks
			indices = np.random.choice(range(len(St_history)), size=batch)       # Get S, S2, St, R, A indices of each frame

			# Sample batch size from replay buffer
			S_sample = np.array([S_history[i] for i in indices])
			S2_sample = np.array([S2_history[i] for i in indices])
			R_sample = [R_history[i] for i in indices]
			A_sample = [A_history[i] for i in indices]
			St_sample = tf.convert_to_tensor([float(St_history[i]) for i in indices])

			# Train target network
			R2 = QT.predict(S2_sample)                                           # Try to predict future reward R2 from S2
			new_q_values = R_sample + Y * tf.reduce_max(R2, axis=1)              # An entire back of R and R2 to calculate a batch of q_values
			new_q_values = new_q_values * (1 - St_sample) - St_sample            # ??? If final frame set the last value to -1
			masks = tf.one_hot(A_sample, actions)                                # Make a mask on only the updated q_values
			with tf.GradientTape() as tape:
				q_values = QP(S_sample)                                          # Get q_value predictions from prediction neural network
				q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)   # Get actions taken for each q_value
				loss = loss_function(new_q_values, q_action)                     # Get loss between old and new q_values

			# Backpropagation
			gradients = tape.gradient(loss, QP.trainable_variables)
			optimizer.apply_gradients(zip(gradients, QT.trainable_variables))

			# Update target neural network
			if frame % update_QT == 0:                                           # Every 10,000 n steps
				QT.set_weights(QP.get_weights())                                 # Make the QP network weights = the QT network weights
				template = 'running reward: {:.2f} at episode {}, frame count {}'
				print(template.format(Reward, episode, frame))

			# Limit replay buffer memory size
			if len(R_history) > N:
				del R_history[:1]
				del S_history[:1]
				del S2_history[:1]
				del A_history[:1]
				del St_history[:1]

            if St:                                                               # End trajectory if St becomes True
            	break

	# Episode wide updates
	Gt_history.append(Gt)                                                        # Append to Gt history
	if len(Gt_history) > 100: del Gt_history[:1]                                 # Limit memory size
	Reward = np.mean(Gt_history)                                                 # Average Gt
	episode += 1                                                                 # Increment episode number

	if Reward > 40:                                                              # Condition to consider the task solved: if average Gt = 40
		print('Solved at episode {}!'.format(episode))
		break
