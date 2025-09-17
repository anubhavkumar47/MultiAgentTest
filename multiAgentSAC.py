import gymnasium as gym
from gymnasium.spaces import Box
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
import os
import copy
from mpl_toolkits.mplot3d import Axes3D
import math

# --- Define a global device for consistency ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REPLACED: New Prioritized Replay Buffer ---
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device="cpu", prioritized=False, alpha=0.6):
        self.capacity = capacity
        self.device = device
        self.prioritized = prioritized
        self.alpha = alpha  # prioritization strength

        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # For prioritized replay
        if prioritized:
            self.priorities = np.zeros((capacity,), dtype=np.float32)
            self.eps = 1e-6  # small constant to avoid zero probability

        # Pointers
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        
        self.states[self.ptr] = state.cpu().numpy()
        self.actions[self.ptr] = action.cpu().numpy()
        self.rewards[self.ptr] = reward.cpu().numpy()
        self.next_states[self.ptr] = next_state.cpu().numpy()
        self.dones[self.ptr] = done.cpu().numpy()

        if self.prioritized:
            # Assign max priority initially so new samples are likely to be chosen
            max_prio = self.priorities.max() if self.size > 0 else 1.0
            self.priorities[self.ptr] = max_prio

        # Update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """Sample batch of transitions"""
        if self.prioritized:
            # Compute probabilities
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(self.size, batch_size, p=probs)
            # Importance-sampling weights
            weights = (self.size * probs[indices]) ** (-beta)
            weights /= weights.max()  # normalize
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            weights = torch.ones((batch_size, 1), device=self.device)

        # Convert to torch tensors directly
        return (
            torch.tensor(self.states[indices], device=self.device),
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(self.rewards[indices], device=self.device),
            torch.tensor(self.next_states[indices], device=self.device),
            torch.tensor(self.dones[indices], device=self.device),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities after learning step"""
        if self.prioritized:
            self.priorities[indices] = np.abs(td_errors.detach().cpu().numpy()) + self.eps

    def __len__(self):
        return self.size


# --- NEW: SAC Actor Network (Replaces DiffusionActor) ---
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1.0):
        super(GaussianActor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Action rescaling
        self.action_scale = torch.tensor((max_action), device=DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean

# --- Centralized Transformer Critic Network (Unchanged) ---
class TransformerCritic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim):
        super(TransformerCritic, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        self.input_embed1 = nn.Linear(input_dim, hidden_dim)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=2)
        self.output_head1 = nn.Linear(hidden_dim * num_agents, 1)
        self.input_embed2 = nn.Linear(input_dim, hidden_dim)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=2)
        self.output_head2 = nn.Linear(hidden_dim * num_agents, 1)

    def forward(self, states, actions):
        batch_size = states.shape[0]
        states_expanded = states.unsqueeze(1).expand(-1, self.num_agents, -1)
        actions_reshaped = actions.view(batch_size, self.num_agents, self.action_dim)
        x = torch.cat([states_expanded, actions_reshaped], dim=-1)
        x1 = self.input_embed1(x)
        x1 = self.transformer_encoder1(x1)
        x1 = x1.view(batch_size, -1)
        q1 = self.output_head1(x1)
        x2 = self.input_embed2(x)
        x2 = self.transformer_encoder2(x2)
        x2 = x2.view(batch_size, -1)
        q2 = self.output_head2(x2)
        return q1, q2

# --- LLM Simulator for High-Level Decisions (Unchanged) ---
class LLM_Simulator:
    def __init__(self):
        self.base_aoi_penalty_weight = 0.01
        self.base_propulsion_penalty_weight = 0.1

    def adjust_penalties(self, avg_aoi, uav_propulsion_energy):
        aoi_weight = self.base_aoi_penalty_weight
        energy_weight = self.base_propulsion_penalty_weight
        if avg_aoi > 40: aoi_weight *= 5
        if uav_propulsion_energy > 30: energy_weight *= 3
        return aoi_weight, energy_weight

    def decide_task_offload(self, uav_positions, iotd_positions):
        num_collectors = 2
        distances = torch.cdist(uav_positions[:num_collectors], iotd_positions)
        iot_indices = torch.sort(distances.flatten()).indices
        assignments, assigned_uavs, assigned_iots = [], set(), set()
        for idx in iot_indices:
            uav_idx = idx // iotd_positions.shape[0]
            iot_idx = idx % iotd_positions.shape[0]
            if uav_idx.item() not in assigned_uavs and iot_idx.item() not in assigned_iots:
                assignments.append((uav_idx.item(), iot_idx.item()))
                assigned_uavs.add(uav_idx.item())
                assigned_iots.add(iot_idx.item())
            if len(assignments) == num_collectors: break
        return assignments

# --- NEW: Multi-Agent SAC Agent ---
class MASAC:
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim, max_action, args):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.target_update_interval = args['target_update_interval']
        self.total_it = 0

        # Actors
        self.actors = [GaussianActor(state_dim, action_dim, hidden_dim, max_action).to(self.device) for _ in range(num_agents)]
        self.actors_optimizer = [optim.Adam(actor.parameters(), lr=args['lr']) for actor in self.actors]

        # Critic
        self.critic = TransformerCritic(num_agents, state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args['lr'])

        # Automatic Entropy Tuning
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(DEVICE)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=args['lr'])


    def select_action(self, states, evaluate=False):
        states = [torch.FloatTensor(s).to(self.device).unsqueeze(0) for s in states]
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                if evaluate:
                    _, _, action = self.actors[i].sample(states[i])
                else:
                    action, _, _ = self.actors[i].sample(states[i])
                actions.append(action.cpu().data.numpy().flatten())
        return np.array(actions)

    def update_parameters(self, memory, beta=0.4):
        self.total_it += 1
        if len(memory) < config['batch_size']:
            return None, None, None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = memory.sample(config['batch_size'], beta)
        
        # Normalization
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        state_batch = (state_batch - state_batch.mean()) / (state_batch.std() + 1e-8)
        next_state_batch = (next_state_batch - next_state_batch.mean()) / (next_state_batch.std() + 1e-8)
        action_batch = (action_batch - action_batch.mean()) / (action_batch.std() + 1e-8)

        # --- Critic Update ---
        with torch.no_grad():
            next_actions = []
            next_log_pis = []
            for i in range(self.num_agents):
                next_action, next_log_pi, _ = self.actors[i].sample(next_state_batch)
                next_actions.append(next_action)
                next_log_pis.append(next_log_pi)
            
            next_actions = torch.cat(next_actions, dim=1)
            next_log_pis = torch.cat(next_log_pis, dim=1).sum(dim=1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * target_Q

        qf1, qf2 = self.critic(state_batch, action_batch)
        
        # Weighted loss for prioritized replay
        qf1_loss = F.mse_loss(qf1, next_q_value, reduction='none')
        qf2_loss = F.mse_loss(qf2, next_q_value, reduction='none')
        qf_loss = (weights * (qf1_loss + qf2_loss)).mean()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Update priorities in replay buffer
        td_errors = (torch.min(qf1_loss, qf2_loss)).detach().squeeze()
        memory.update_priorities(indices, td_errors)
        
        # --- Actor and Alpha Update ---
        actions_pred, log_pis_pred = [], []
        for i in range(self.num_agents):
            pi, log_pi, _ = self.actors[i].sample(state_batch)
            actions_pred.append(pi)
            log_pis_pred.append(log_pi)

        actions_pred = torch.cat(actions_pred, dim=1)
        log_pis_pred = torch.cat(log_pis_pred, dim=1).sum(dim=1, keepdim=True)
        
        qf1_pi, qf2_pi = self.critic(state_batch, actions_pred)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pis_pred) - min_qf_pi).mean()

        for optim in self.actors_optimizer: optim.zero_grad()
        policy_loss.backward()
        for optim in self.actors_optimizer: optim.step()
        
        # Alpha (temperature) update
        alpha_loss = -(self.log_alpha * (log_pis_pred + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()
        
        # --- Soft Target Updates ---
        if self.total_it % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return qf_loss.item(), policy_loss.item(), alpha_tlogs.item()


# --- Custom Multi-Agent Environment (Unchanged) ---
class MultiAgentEnv(gym.Env):
    def __init__(self, num_uavs=3, num_IoTD=5, T=100):
        super().__init__()
        self.device = DEVICE
        self.num_agents = self.num_uavs = num_uavs
        self.num_collectors, self.num_IoTD, self.T = 2, num_IoTD, T
        self.size = torch.tensor([300.0, 300.0, 300.0], device=self.device)
        iot_positions = np.array([[50,50,0],[75,150,0],[100,100,0],[100,250,0],[150,150,0]], dtype=np.float32)
        self.iotd_position = torch.tensor(iot_positions, device=self.device, dtype=torch.float32)
        self.eavesdropper_pos = torch.tensor([150.0, 0.0, 0.0], device=self.device)
        self.R_min, self.P_tx_UAV, self.P_tx_IoTD, self.P_jammer = 0.1, 0.5, 0.1, 0.1
        self.eta, self.beta_0, self.noise_power = 0.5, 1e-3, 1e-13
        self.collision_threshold = 5.0
        self.iot_idle_drain_rate = 0.001
        self.iot_comm_drain_rate = 0.05
        self.uav_comm_energy_cost = 0.02
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_shape = (self.num_uavs * 3) + (2 * self.num_IoTD) + self.num_uavs
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.llm = LLM_Simulator()
        self.current_assignments = []
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None: super().reset(seed=seed)
        low = torch.tensor([0.0, 0.0, 100.0], device=self.device)
        self.uav_positions = low + torch.rand((self.num_uavs, 3), device=self.device) * (self.size - low)
        self.AoI = torch.zeros(self.num_IoTD, device=self.device)
        self.iot_energy_levels = torch.ones(self.num_IoTD, device=self.device)
        self.uav_energy_levels = torch.ones(self.num_uavs, device=self.device)
        self.time = 0
        self.collision_count = 0
        return self._get_obs(), {}

    def _get_global_state(self):
        return torch.cat([self.uav_positions.flatten(), self.AoI, self.iot_energy_levels, self.uav_energy_levels])

    def _get_obs(self):
        global_state = self._get_global_state().detach().cpu().numpy()
        return [global_state for _ in range(self.num_agents)]

    def step(self, actions):
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.float32)
        self.time += 1
        self.AoI += 1.0
        self.iot_energy_levels -= self.iot_idle_drain_rate
        self.iot_energy_levels.clamp_(min=0.0)
        past_positions = self.uav_positions.clone()
        self.uav_positions += actions_t * 20.0
        collision_penalty = self._check_collisions_torch()
        boundary_penalty = self._check_boundaries_torch()
        propulsion_energy = torch.sum(torch.linalg.norm(self.uav_positions - past_positions, dim=1))
        avg_aoi = torch.mean(self.AoI).item()
        aoi_penalty_weight, energy_penalty_weight = self.llm.adjust_penalties(avg_aoi, propulsion_energy.item())
        if self.time % 25 == 0 or not self.current_assignments:
            self.current_assignments = self.llm.decide_task_offload(self.uav_positions, self.iotd_position)
        reward_success, reward_fail = 0.0, 0.0
        for uav_idx, iot_idx in self.current_assignments:
            if self.uav_energy_levels[uav_idx] < self.uav_comm_energy_cost:
                reward_fail -= 0.5; continue
            collector_uav_pos = self.uav_positions[uav_idx]
            dist_UAV_IoTD = torch.linalg.norm(collector_uav_pos - self.iotd_position[iot_idx])
            energy_harvested = self.P_tx_UAV * (self.beta_0 / (dist_UAV_IoTD**2 + 1e-9)) * self.eta
            self.iot_energy_levels[iot_idx] += energy_harvested
            if self.iot_energy_levels[iot_idx] >= self.iot_comm_drain_rate:
                self.iot_energy_levels[iot_idx] -= self.iot_comm_drain_rate
                secure_rate = self._calculate_secure_rate_torch(collector_uav_pos, self.iotd_position[iot_idx])
                if secure_rate > self.R_min:
                    reward_success += 10.0; self.AoI[iot_idx] = 0.0
                else: reward_fail -= 1.0
            else: reward_fail -= 1.0
            self.uav_energy_levels[uav_idx] -= self.uav_comm_energy_cost
        self.iot_energy_levels.clamp_(min=0.0, max=1.0)
        self.uav_energy_levels.clamp_(min=0.0, max=1.0)
        aoi_penalty = -aoi_penalty_weight * torch.mean(self.AoI).item()
        energy_penalty = -energy_penalty_weight * propulsion_energy.item()
        reward = reward_success + reward_fail + aoi_penalty + energy_penalty + collision_penalty + boundary_penalty
        terminated = self.time >= self.T
        info = {'propulsion_energy': propulsion_energy.item(), 'sum_AoI': torch.sum(self.AoI).item(), 'collisions': self.collision_count, 'avg_iot_energy': torch.mean(self.iot_energy_levels).item(), 'avg_uav_energy': torch.mean(self.uav_energy_levels).item()}
        return self._get_obs(), reward, terminated, False, info

    def _check_boundaries_torch(self):
        lower, upper = torch.zeros(3, device=self.device), self.size
        out_of_bounds = torch.any((self.uav_positions < lower) | (self.uav_positions > upper), dim=1)
        penalty = -1.0 * torch.sum(out_of_bounds).item()
        self.uav_positions.clamp_(lower.expand_as(self.uav_positions), upper.expand_as(self.uav_positions))
        return penalty

    def _check_collisions_torch(self):
        penalty = 0.0
        for uav1_idx, uav2_idx in combinations(range(self.num_uavs), 2):
            dist = torch.linalg.norm(self.uav_positions[uav1_idx] - self.uav_positions[uav2_idx])
            if dist < self.collision_threshold:
                penalty -= 1.0; self.collision_count += 1
        return penalty

    def _calculate_secure_rate_torch(self, collector_pos, iotd_pos):
        jammer_pos = self.uav_positions[2]
        get_gain = lambda p1, p2: self.beta_0 / (torch.sum((p1-p2)**2) + 1e-9)
        sig_main = self.P_tx_IoTD * get_gain(iotd_pos, collector_pos)
        inter_main = self.P_jammer * get_gain(jammer_pos, collector_pos) + self.noise_power
        rate_main = torch.log2(1.0 + sig_main / inter_main)
        sig_eve = self.P_tx_IoTD * get_gain(iotd_pos, self.eavesdropper_pos)
        inter_eve = self.P_jammer * get_gain(jammer_pos, self.eavesdropper_pos) + self.noise_power
        rate_eve = torch.log2(1.0 + sig_eve / inter_eve)
        return torch.clamp(rate_main - rate_eve, min=0.0)

# --- Plotting and Visualization (Unchanged) ---
def plot_and_save_results(log_df, filename="masac_training_performance.png"):
    fig, axs = plt.subplots(7, 1, figsize=(12, 35), sharex=True)
    fig.suptitle('MASAC with Gaussian Actor Training Performance', fontsize=18)
    axs[0].plot(log_df['episode'], log_df['reward'], color='green'); axs[0].set_title("Episodic Reward"); axs[0].set_ylabel("Reward"); axs[0].grid(True)
    axs[1].plot(log_df['episode'], log_df['avg_critic_loss'], color='red'); axs[1].set_title("Average Critic Loss"); axs[1].set_ylabel("Loss"); axs[1].grid(True)
    axs[2].plot(log_df['episode'], log_df['avg_actor_loss'], color='blue'); axs[2].set_title("Average Actor Loss"); axs[2].set_ylabel("Loss"); axs[2].grid(True)
    axs[3].plot(log_df['episode'], log_df['avg_propulsion_energy'], color='purple'); axs[3].set_title("Average Propulsion Energy"); axs[3].set_ylabel("Energy"); axs[3].grid(True)
    axs[4].plot(log_df['episode'], log_df['avg_sum_aoi'], color='orange'); axs[4].set_title("Average Sum of AoI"); axs[4].set_ylabel("AoI"); axs[4].grid(True)
    axs[5].plot(log_df['episode'], log_df['avg_uav_energy'], color='cyan', label='Avg UAV Energy')
    axs[5].plot(log_df['episode'], log_df['avg_iot_energy'], color='magenta', label='Avg IoT Energy')
    axs[5].set_title("Average Energy Levels"); axs[5].set_ylabel("Energy Level (0-1)"); axs[5].grid(True); axs[5].legend()
    axs[6].plot(log_df['episode'], log_df['total_collisions'], color='brown'); axs[6].set_title("Total Collisions per Episode"); axs[6].set_ylabel("Count"); axs[6].set_xlabel("Episode"); axs[6].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.savefig(filename); plt.close()
    print(f"\nPlots saved to {os.path.abspath(filename)}")

def visualize_episode(episode_history, env_params, filename="uav_flight_paths_sac.png"):
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    iot_pos = env_params['iotd_position'].cpu().numpy(); eve_pos = env_params['eavesdropper_pos'].cpu().numpy()
    ax.scatter(iot_pos[:,0],iot_pos[:,1],iot_pos[:,2], c='blue', marker='o', s=100, label='IoT Devices')
    ax.scatter(eve_pos[0], eve_pos[1], eve_pos[2], c='red', marker='x', s=150, label='Eavesdropper')
    num_uavs = episode_history[0]['uav_positions'].shape[0]
    uav_trajectories = [[] for _ in range(num_uavs)]
    for step_data in episode_history:
        for i in range(num_uavs): uav_trajectories[i].append(step_data['uav_positions'][i])
    colors = ['orange', 'magenta', 'yellow']
    for i in range(num_uavs):
        path = np.array(uav_trajectories[i])
        ax.plot(path[:,0], path[:,1], path[:,2], color=colors[i], label=f'UAV {i+1} Path')
        ax.scatter(path[0,0],path[0,1],path[0,2], color='green', marker='s', s=100, label=f'UAV {i+1} Start' if i==0 else "")
        ax.scatter(path[-1,0],path[-1,1],path[-1,2], color='black', marker='*', s=150, label=f'UAV {i+1} End' if i==0 else "")
    final_assignments = episode_history[-1]['assignments']; final_uav_pos = episode_history[-1]['uav_positions']
    for uav_idx, iot_idx in final_assignments:
        uav_p, iot_p = final_uav_pos[uav_idx], iot_pos[iot_idx]
        ax.plot([uav_p[0], iot_p[0]], [uav_p[1], iot_p[1]], [uav_p[2], iot_p[2]], 'k--')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (Altitude)'); ax.set_title('UAV Flight Paths')
    size = env_params['size'].cpu().numpy()
    ax.set_xlim(0, size[0]); ax.set_ylim(0, size[1]); ax.set_zlim(0, size[2])
    ax.legend(); ax.grid(True); plt.savefig(filename); plt.close()
    print(f"\nEpisode visualization saved to {os.path.abspath(filename)}")

# --- MODIFIED: Constants and Hyperparameters for SAC ---
config = {
    'seed': 12345, 'max_episodes': 3000, 'replay_size': 100000, 'gamma': 0.99,
    'tau': 0.005, 'lr': 3e-4, 'hidden_size': 256, 'batch_size': 128, 'start_steps': 2000,
    'alpha': 0.2, # Initial alpha value (temperature)
    'target_update_interval': 1, # How often to update target networks
}

def main():
    env = MultiAgentEnv()
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])

    # --- MODIFIED: Instantiate MASAC agent ---
    agent = MASAC(
        num_agents=env.num_agents, 
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0], 
        hidden_dim=config['hidden_size'],
        max_action=env.action_space.high[0], 
        args=config
    )

    # Replay buffer initialization remains the same
    total_action_dim = env.num_agents * env.action_space.shape[0]
    memory = ReplayBuffer(
        state_dim=env.observation_space.shape[0],
        action_dim=total_action_dim,
        capacity=config['replay_size'],
        device=DEVICE,
        prioritized=True 
    )
    
    total_numsteps = 0; training_logs = []

    # Parameters for Prioritized Replay Beta Annealing
    beta_start = 0.4
    beta_frames = config['max_episodes'] * env.T
    beta = beta_start

    for i_episode in range(config['max_episodes']):
        obs, _ = env.reset(seed=config['seed'] + i_episode)
        episode_reward, episode_steps = 0, 0; done = False
        episode_critic_losses, episode_actor_losses, episode_alpha_losses = [], [], []
        episode_propulsion_energy, episode_sum_aoi = [], []
        episode_uav_energy, episode_iot_energy, episode_collisions = [], [], 0

        while not done:
            if config['start_steps'] > total_numsteps:
                actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            else:
                actions = agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Push to replay buffer (unchanged)
            flat_state = torch.tensor(obs[0], dtype=torch.float32)
            flat_next_state = torch.tensor(next_obs[0], dtype=torch.float32)
            flat_actions = torch.tensor(actions.reshape(-1), dtype=torch.float32)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.float32)
            memory.push(flat_state, flat_actions, reward_tensor, flat_next_state, done_tensor)

            if total_numsteps > config['batch_size']*5:
                # Update beta for PER
                beta = min(1.0, beta_start + total_numsteps * (1.0 - beta_start) / beta_frames)
                # --- MODIFIED: Call MASAC update ---
                critic_loss, actor_loss, alpha_loss = agent.update_parameters(memory, beta)
                if critic_loss is not None: episode_critic_losses.append(critic_loss)
                if actor_loss is not None: episode_actor_losses.append(actor_loss)
                if alpha_loss is not None: episode_alpha_losses.append(alpha_loss)

            episode_propulsion_energy.append(info['propulsion_energy'])
            episode_sum_aoi.append(info['sum_AoI'])
            episode_uav_energy.append(info['avg_uav_energy'])
            episode_iot_energy.append(info['avg_iot_energy'])
            episode_collisions = info['collisions']
            obs = next_obs; episode_steps += 1; total_numsteps += 1; episode_reward += reward

        critic_loss_mean = np.mean(episode_critic_losses) if episode_critic_losses else -1
        actor_loss_mean = np.mean(episode_actor_losses) if episode_actor_losses else -1
        log_entry = {
            'episode': i_episode + 1, 'reward': episode_reward/100, 
            'avg_critic_loss': critic_loss_mean, 'avg_actor_loss': actor_loss_mean, 
            'avg_propulsion_energy': np.mean(episode_propulsion_energy),
            'avg_sum_aoi': np.mean(episode_sum_aoi),
            'avg_uav_energy': np.mean(episode_uav_energy),
            'avg_iot_energy': np.mean(episode_iot_energy),
            'total_collisions': episode_collisions
        }
        training_logs.append(log_entry)
        print(f"E: {i_episode+1}, R: {episode_reward/100:.2f}, EGY: {np.mean(episode_propulsion_energy):.2f}, AoI: {np.mean(episode_sum_aoi):.2f}, CL: {critic_loss_mean:.4f}, AL: {actor_loss_mean:.4f}")

    env.close()
    log_df = pd.DataFrame(training_logs)
    log_df.to_csv('masac_training_logs.csv', index=False)
    plot_and_save_results(log_df)

    print("\nRunning final episode for visualization...")
    obs, _ = env.reset(); done = False; episode_history = []
    while not done:
        actions = agent.select_action(obs, evaluate=True) # Use deterministic actions for visualization
        next_obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        episode_history.append({'uav_positions': env.uav_positions.cpu().numpy(), 'assignments': env.current_assignments})
        obs = next_obs
    visualize_episode(episode_history, {'iotd_position':env.iotd_position, 'eavesdropper_pos':env.eavesdropper_pos, 'size':env.size})

if __name__ == '__main__':
    main()
