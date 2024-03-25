# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import torch
import random
import numpy as np
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

env_id                   = 'CartPole-v1'
seed                     = 1
num_envs                 = 1
learning_rate            = 2.5e-4
buffer_size              = 10000
batch_size               = 128
total_timesteps          = 500000
start_e                  = 1
end_e                    = 0.05
exploration_fraction     = 0.5
learning_starts          = 10000
train_frequency          = 10
gamma                    = 0.99
target_network_frequency = 500
tau                      = 1

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class QNetwork(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, env.single_action_space.n),)
    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# Seeding
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
# Environment setup
envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i, i, False, 'X') for i in range(num_envs)])
q_network = QNetwork(envs).to(device)
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict()) 
rb = ReplayBuffer(
    buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False)
# Start the game
obs, _ = envs.reset(seed=seed)
for global_step in range(total_timesteps):
    # Action logic
    epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
    # Play the game and log data
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)
    # Rewards
    if 'final_info' in infos:
        for info in infos['final_info']:
            if 'episode' not in info:
                continue
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
    # Save data to reply buffer
    real_next_obs = next_obs.copy()
    for idx, d in enumerate(truncated):
        if d:
            real_next_obs[idx] = infos['final_observation'][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
    obs = next_obs
    # Training
    if global_step > learning_starts:
        if global_step % train_frequency == 0:
            data = rb.sample(batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = torch.nn.functional.mse_loss(td_target, old_val)
            # Backpropagation and gradient discent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Update target network
        if global_step % target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(tau * q_network_param.data + (1.0 - tau) * target_network_param.data)
