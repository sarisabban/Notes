# pip install numpy matplotlib gymnasium torch
# his algorithm is a derivation from CleanRL:
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# https://youtu.be/MEt6rrxH8W4

import torch
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env         = 'CartPole-v1'
lr          = 2.5e-4
seed        = 1
timesteps   = 50000
n_envs      = 4
n_steps     = 128
epochs      = 4
minibatches = 4
batches     = n_envs * n_steps
n_updates   = timesteps // batches
gamma       = 0.99
lambd       = 0.95
clip        = 0.2
ent         = 0.01
vf_coef     = 0.5
g_norm      = 0.5
target_kl   = 0.015

# This environment setup is very important
def make_env(env_id):
	def thunk():
		env = gym.make(env_id)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		return env
	return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer

class Agent(torch.nn.Module):
	''' The PPO network '''
	def __init__(self, envs):
		super(Agent, self).__init__()
		obs_shape = envs.single_observation_space.shape
		act_shape = envs.single_action_space.n
		self.actor = torch.nn.Sequential(
			layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
			torch.nn.Tanh(),
			layer_init(torch.nn.Linear(64, 64)),
			torch.nn.Tanh(),
			torch.nn.Flatten(),
			layer_init(torch.nn.Linear(64, act_shape), std=0.01))
		self.critic = torch.nn.Sequential(
			layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
			torch.nn.Tanh(),
			layer_init(torch.nn.Linear(64, 64)),
			torch.nn.Tanh(),
			torch.nn.Flatten(),
			layer_init(torch.nn.Linear(64, 1), std=1.0))
	def get_value(self, x):
		return self.critic(x)
	def get_action_and_value(self, x, action=None):
		logits = self.actor(x)
		probs  = torch.distributions.categorical.Categorical(logits=logits)
		if action is None: action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# Fix seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training on:', device)
# Vectorise the environment
envs = gym.vector.SyncVectorEnv([make_env(env) for i in range(n_envs)])
# Call agent
agent = Agent(envs).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
# Dataset batch size buffer
s_obs_space = envs.single_observation_space.shape
s_act_space = envs.single_action_space.shape
obs         = torch.zeros((n_steps, n_envs) + s_obs_space).to(device)
actions     = torch.zeros((n_steps, n_envs) + s_act_space).to(device)
rewards     = torch.zeros((n_steps, n_envs)).to(device)
values      = torch.zeros((n_steps, n_envs)).to(device)
dones       = torch.zeros((n_steps, n_envs)).to(device)
Plogs       = torch.zeros((n_steps, n_envs)).to(device)
next_obs    = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
next_done   = torch.zeros(n_envs).to(device)
# Training
global_step = 0
Us, Ts, Rs, As, Cs, Es, Ls, Ks, Cl, Var = [], [], [], [], [], [], [], [], [], []
for update in range(1, n_updates + 1):
	# Anneal learning rate
	optimizer.param_groups[0]['lr'] = (1.0 - (update - 1.0) / n_updates) * lr
	# Collect a batch of data
	for step in range(n_steps):
		global_step += 1 * n_envs
		obs[step] = next_obs
		dones[step] = next_done
		with torch.no_grad():
			A, Plog, E, V = agent.get_action_and_value(next_obs)
			values[step] = V.flatten()
		actions[step] = A
		Plogs[step] = Plog
		# Play game
		########################################################################
		next_obs, R, T, U, info = envs.step(A.cpu().numpy())
		done = T + U
		rewards[step] = torch.tensor(R).to(device).view(-1)
		next_obs      = torch.Tensor(next_obs).to(device)
		next_done     = torch.Tensor(done).to(device)
		if 'final_info' in info.keys():
			for inf in info['final_info']:
				if type(inf) is dict:
					Gt = inf['episode']['r'][0]
					Rs.append(Gt)
					Ts.append(global_step)
					print(f'Steps: {global_step:<10,} Returns: {Gt}')
					break
		########################################################################
	# Bootstra value (GAE method)
	with torch.no_grad():
		next_value = agent.get_value(next_obs).reshape(1, -1)
		advantages = torch.zeros_like(rewards).to(device)
		lastgaelam = 0
		for t in reversed(range(n_steps)):
			if t == n_steps - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - dones[t + 1]
				nextvalues = values[t + 1]
			delta = \
			rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
			advantages[t] = lastgaelam = \
			delta + gamma * lambd * nextnonterminal * lastgaelam
		returns = advantages + values
	# Flatten the batch
	b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
	b_Plog       = Plogs.reshape(-1)
	b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
	b_advantages = advantages.reshape(-1)
	b_returns    = returns.reshape(-1)
	b_values     = values.reshape(-1)
	# Training
	b_indx = np.arange(batches)
	clipfracs = []
	for epoch in range(epochs):
		np.random.shuffle(b_indx)
		for start in range(0, batches, minibatches):
			# Get a mini batch index segment
			end = start + minibatches
			mb_indx = b_indx[start:end]
			# Use St and At -> qt+1
			_, newPlog, newE, newV = agent.get_action_and_value(b_obs[mb_indx], b_actions.long()[mb_indx])
			# Calculate the ratio between the new and old probabilities (Plog)
			ratiolog = newPlog - b_Plog[mb_indx]
			ratio    = ratiolog.exp()
			# Normalise the advantage
			mb_advantages = b_advantages[mb_indx]
			mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8)
			# Objective clipping
			p_loss1 = -mb_advantages * ratio
			p_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
			p_loss = torch.max(p_loss1, p_loss2).mean()
			# Value loss
			newV = newV.view(-1)
			v_loss_unclipped = (newV - b_returns[mb_indx]) ** 2
			v_clipped = b_values[mb_indx] + torch.clamp(
				newV - b_values[mb_indx], -clip, clip)
			v_loss_clipped = (v_clipped - b_returns[mb_indx]) ** 2
			v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
			v_loss = 0.5 * v_loss_max.mean()
			# Entropy loss
			e_loss = newE.mean()
			# Final loss
			loss = p_loss - ent * e_loss + v_loss * vf_coef
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(agent.parameters(), g_norm)
			optimizer.step()
			# Aproximate KL divergence
			with torch.no_grad():
				approx_kl = ((ratio - 1) - ratiolog).mean()
				clipfracs += [((ratio - 1.0).abs() > clip).float().mean().item()]
		Cl.append(len([x for x in clipfracs if x != 0.0]))
		y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
		var_y = np.var(y_true)
		explained_var = np.nan if var_y == 0 else 1-np.var(y_true - y_pred) / var_y
		# Early stopping
		if target_kl is not None:
			if approx_kl > target_kl: break
		Us.append(update)
		As.append(p_loss.item())
		Cs.append(v_loss.item())
		Es.append(e_loss.item())
		Ls.append(loss.item())
		Ks.append(approx_kl.item())
		Var.append(explained_var.item())

# Plot training results
fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(Ts, Rs)
axs[0, 0].set_title('Returns')
axs[0, 0].set(xlabel='Steps', ylabel='Returns')

axs[0, 1].plot(Us, Ls)
axs[0, 1].set_title('Final Loss')
axs[0, 1].set(xlabel='Updates', ylabel='Loss')

axs[1, 1].plot(Us, As)
axs[1, 1].set_title('Actor loss')
axs[1, 1].set(xlabel='Steps', ylabel='Loss')

axs[2, 1].plot(Us, Cs)
axs[2, 1].set_title('Critic loss')
axs[2, 1].set(xlabel='Steps', ylabel='Loss')

axs[3, 1].plot(Us, Es)
axs[3, 1].set_title('Entropy loss')
axs[3, 1].set(xlabel='Steps', ylabel='Loss')

axs[1, 0].plot(Us, Ks)
axs[1, 0].set_title('KL aproximation')
axs[1, 0].set(xlabel='Steps', ylabel='Ratio')

axs[3, 0].plot(Us, Var)
axs[3, 0].set_title('Explained variance')
axs[3, 0].set(xlabel='Steps', ylabel='variance')

if len(Us) != len(Cl): Us.insert(0, 0)
axs[2, 0].plot(Us, Cl)
axs[2, 0].set_title('Clip fractions')
axs[2, 0].set(xlabel='Steps', ylabel='Fractions')

plt.show()
