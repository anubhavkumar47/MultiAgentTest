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

# --- Constants and Hyperparameters ---
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Use CUDA if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'seed': 42,
    'start_steps': 2500,
    'max_episodes': 1000,
    'replay_size': 100000,
    'gamma': 0.99,
    'tau': 0.001,
    'alpha': 0.2,
    'lr_actor': 0.0004,   # Learning rate for the actor networks
    'lr_critic': 0.0003,  # Learning rate for the critic network
    'lr_alpha': 0.0001,   # Learning rate for the entropy temperature
    'hidden_size': 256,
    'batch_size': 256,
    'epsilon': 0.1,         # 10% random exploration
    'epsilon_decay': 0.995, # optional: decay per episode
    'epsilon_min': 0.01,     # minimum epsilon
    'automatic_entropy_tuning': True
}

# --- Replay Buffer ---
class ReplayBuffer:
    """A replay buffer for storing multi-agent experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, global_state, actions, reward, next_global_state, done):
        """Saves a transition for the entire multi-agent system."""
        self.buffer.append((global_state, actions, reward, next_global_state, done))

    def sample(self, batch_size):
        global_state, actions, reward, next_global_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(global_state), np.array(actions), np.array(reward), np.array(next_global_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- LLM Simulator for High-Level Decisions ---
class LLM_Simulator:
    """
    A simulated LLM to make high-level decisions like adjusting penalties
    and handling task offloading based on the current environment state.
    """
    def __init__(self):
        self.base_aoi_penalty_weight = 0.01
        self.base_propulsion_penalty_weight = 0.1
        print("LLM Simulator Initialized: Ready to provide strategic guidance.")

    def adjust_penalties(self, avg_aoi, uav_propulsion_energy):
        """Dynamically adjusts penalty weights based on situational context."""
        aoi_weight = self.base_aoi_penalty_weight
        energy_weight = self.base_propulsion_penalty_weight

        # Situation 1: AoI is becoming critically high
        if avg_aoi > 40:
            aoi_weight *= 5  # Increase penalty to prioritize AoI reduction

        # Situation 2: High energy consumption detected
        if uav_propulsion_energy > 30: # Assuming 30 is a high value
            energy_weight *= 3 # Increase penalty to encourage more efficient flight

        return aoi_weight, energy_weight

    def decide_task_offload(self, uav_positions, iotd_positions):
        """
        Simulates task offloading by re-assigning collectors to the nearest IoT devices.
        This represents a high-level strategic decision.
        """
        num_collectors = 2
        distances = torch.cdist(uav_positions[:num_collectors], iotd_positions)
        
        assignments = []
        assigned_iots = set()
        
        sorted_distances = []
        for uav_idx in range(num_collectors):
            for iot_idx in range(iotd_positions.shape[0]):
                sorted_distances.append((distances[uav_idx, iot_idx].item(), uav_idx, iot_idx))
        
        sorted_distances.sort()

        assigned_uavs = set()
        for _, uav_idx, iot_idx in sorted_distances:
            if uav_idx not in assigned_uavs and iot_idx not in assigned_iots:
                assignments.append((uav_idx, iot_idx))
                assigned_uavs.add(uav_idx)
                assigned_iots.add(iot_idx)
            if len(assignments) == num_collectors:
                break
        
        return assignments

# --- Decentralized Actor Network ---
class Actor(nn.Module):
    """Gaussian Policy Actor Network for an individual agent."""
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1., device=DEVICE)
            self.action_bias = torch.tensor(0., device=DEVICE)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(DEVICE)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(DEVICE)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- Centralized Critic Network ---
class CentralizedCritic(nn.Module):
    """Twin Q-Network Critic that takes the global state and joint actions."""
    def __init__(self, global_state_dim, joint_action_dim, hidden_dim):
        super(CentralizedCritic, self).__init__()
        input_dim = global_state_dim + joint_action_dim
        # Q1 architecture
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(input_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, joint_action):
        xu = torch.cat([global_state, joint_action], 1)
        x1 = F.relu(self.linear1(xu)); x1 = F.relu(self.linear2(x1)); q1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu)); x2 = F.relu(self.linear5(x2)); q2 = self.linear6(x2)
        return q1, q2

# --- Multi-Agent SAC (MASAC) Agent ---
class MASAC(object):
    def __init__(self, global_state_dim, action_spaces, num_agents, args):
        self.gamma = args['gamma']; self.tau = args['tau']; self.alpha = args['alpha']
        self.batch_size = args['batch_size']
        self.num_agents = num_agents
        self.device = DEVICE
        print(f"Training on device: {self.device}")

        joint_action_dim = sum(space.shape[0] for space in action_spaces)

        self.critic = CentralizedCritic(global_state_dim, joint_action_dim, args['hidden_size']).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['lr_critic'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_update = 0

        # Create decentralized actors with a separate learning rate
        self.actors = []
        self.actor_optims = []
        for i in range(num_agents):
            actor = Actor(global_state_dim, action_spaces[i].shape[0], args['hidden_size'], action_spaces[i]).to(self.device)
            self.actors.append(actor)
            self.actor_optims.append(optim.Adam(actor.parameters(), lr=args['lr_actor']))

        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        if self.automatic_entropy_tuning:
            self.target_entropies = [-torch.prod(torch.Tensor(space.shape).to(self.device)).item() for space in action_spaces]
            self.log_alphas = [torch.zeros(1, requires_grad=True, device=self.device) for _ in range(num_agents)]
            self.alpha_optims = [optim.Adam([log_alpha], lr=args['lr_alpha']) for log_alpha in self.log_alphas]

    def select_actions(self, states, evaluate=False):
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).to(self.device).unsqueeze(0)
            action, _ = self.actors[i].sample(state)
            actions.append(action.detach().cpu().numpy()[0])
        return np.array(actions)

    def update_parameters(self, memory):
        if len(memory) < self.batch_size:
            return None, [None]*self.num_agents

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(1 - done_batch).to(self.device).unsqueeze(1)

        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        state_batch = (state_batch - state_batch.mean()) / (state_batch.std() + 1e-8)
        next_state_batch = (next_state_batch - next_state_batch.mean()) / (next_state_batch.std() + 1e-8)
        action_batch = (action_batch - action_batch.mean()) / (action_batch.std() + 1e-8)
        

        # --- Update Critic ---
        with torch.no_grad():
            next_actions, next_log_pis = [], []
            for i in range(self.num_agents):
                action, log_pi = self.actors[i].sample(next_state_batch)
                next_actions.append(action)
                next_log_pis.append(log_pi)
            
            joint_next_actions = torch.cat(next_actions, dim=1)
            joint_next_log_pi = torch.cat(next_log_pis, dim=1).sum(dim=1, keepdim=True)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, joint_next_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * joint_next_log_pi
            next_q_value = reward_batch + done_batch * self.gamma * min_qf_next_target
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        self.actor_update+=1

        # --- Update Actors ---
        
        policy_losses = []
        if self.actor_update % 5 ==0 :
            for i in range(self.num_agents):
                actions_pred, log_pis = [], []
                for j in range(self.num_agents):
                    if i == j:
                        action, log_pi = self.actors[j].sample(state_batch)
                        actions_pred.append(action)
                        log_pis.append(log_pi)
                    else:
                        # Detach other agents' actions to stabilize learning
                        action_dim = self.actors[j].mean_linear.out_features
                        start_idx = sum(self.actors[k].mean_linear.out_features for k in range(j))
                        end_idx = start_idx + action_dim
                        actions_pred.append(action_batch[:, start_idx:end_idx].detach())
            
                joint_actions_pred = torch.cat(actions_pred, dim=1)
                qf1_pi, qf2_pi = self.critic(state_batch, joint_actions_pred)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
                alpha = self.log_alphas[i].exp().item() if self.automatic_entropy_tuning else self.alpha
                policy_loss = ((alpha * log_pis[0]) - min_qf_pi).mean()
                policy_losses.append(policy_loss.item())

                self.actor_optims[i].zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
                self.actor_optims[i].step()

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alphas[i] * (log_pis[0] + self.target_entropies[i]).detach()).mean()
                    self.alpha_optims[i].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optims[i].step()
        
        self.soft_update(self.critic_target, self.critic, self.tau)
        return qf_loss.item(), policy_losses

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# --- Custom Multi-Agent Environment (Modified) ---
class MultiAgentEnv(gym.Env):
    """
    A custom multi-agent environment for UAV-assisted IoT data collection.
    This modified version includes detailed energy consumption models for both
    UAVs and IoT devices, and a TDMA-based communication protocol.
    """
    def __init__(self, num_uavs=3, num_IoTD=5, T=100):
        super().__init__()
        self.device = DEVICE
        self.num_agents = self.num_uavs = num_uavs
        # In this setup, UAVs 0 and 1 are collectors, UAV 2 is a jammer.
        self.num_collectors, self.num_IoTD, self.T = 2, num_IoTD, T

        # --- Environment Geometry and Positions ---
        self.size = torch.tensor([300.0, 300.0, 300.0], device=self.device)
        iot_positions = np.array([[50,50,0],[75,150,0],[100,100,0],[100,250,0],[150,150,0]], dtype=np.float32)
        self.iotd_position = torch.tensor(iot_positions, device=self.device, dtype=torch.float32)
        self.eavesdropper_pos = torch.tensor([150.0, 0.0, 0.0], device=self.device)

        # --- Physics and Communication Parameters ---
        self.R_min, self.P_tx_UAV, self.P_tx_IoTD, self.P_jammer = 0.2, 0.5, 0.1, 0.1
        self.eta, self.beta_0, self.noise_power = 0.5, 1e-3, 1e-13
        self.collision_threshold = 5.0

        # --- NEW: Energy Consumption Parameters ---
        self.iot_idle_drain_rate = 0.001  # Low constant energy drain for IoT devices
        self.iot_comm_drain_rate = 0.05   # Energy cost for an IoT device to transmit data
        self.uav_comm_energy_cost = 0.02  # Energy cost for a UAV to perform a charge/collect task

        # --- Action and Observation Spaces ---
        self.action_spaces = [Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for _ in range(self.num_uavs)]
        
        # MODIFIED: Observation space shape now includes UAV energy levels
        obs_shape = (self.num_uavs * 3) + (2 * self.num_IoTD) + self.num_uavs
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.llm = LLM_Simulator()
        self.current_assignments = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low = torch.tensor([0.0, 0.0, 100.0], device=self.device)
        self.uav_positions = low + torch.rand((self.num_uavs, 3), device=self.device) * (self.size - low)
        
        # --- State Variables ---
        self.AoI = torch.zeros(self.num_IoTD, device=self.device)
        self.iot_energy_levels = torch.ones(self.num_IoTD, device=self.device)
        # NEW: UAV energy levels are now part of the state
        self.uav_energy_levels = torch.ones(self.num_uavs, device=self.device)
        
        self.time = 0
        self.collision_count = 0
        return self._get_obs(), {}

    def _get_global_state(self):
        """Constructs the global state vector from all environment variables."""
        # MODIFIED: Include UAV energy levels in the state
        return torch.cat([
            self.uav_positions.flatten(), 
            self.AoI, 
            self.iot_energy_levels,
            self.uav_energy_levels
        ]).detach().cpu().numpy()

    def _get_obs(self):
        """Returns a list of observations for each agent (all agents see the global state)."""
        global_state = self._get_global_state()
        return [global_state for _ in range(self.num_agents)]

    def step(self, actions):
        actions_t = torch.from_numpy(actions).to(self.device).float()
        
        self.time += 1
        self.AoI += 1.0
        
        # --- NEW: Apply idle energy drain to all IoT devices at each step ---
        self.iot_energy_levels -= self.iot_idle_drain_rate
        self.iot_energy_levels.clamp_(min=0.0)

        past_positions = self.uav_positions.clone()
        self.uav_positions += actions_t * 20.0

        # Calculate standard penalties and energy costs
        collision_penalty = self._check_collisions_torch()
        boundary_penalty = self._check_boundaries_torch()
        propulsion_energy = torch.sum(torch.linalg.norm(self.uav_positions - past_positions, dim=1))

        # --- LLM Integration ---
        avg_aoi = torch.mean(self.AoI).item()
        aoi_penalty_weight, energy_penalty_weight = self.llm.adjust_penalties(avg_aoi, propulsion_energy.item())

        if self.time % 25 == 0 or not self.current_assignments:
            self.current_assignments = self.llm.decide_task_offload(self.uav_positions, self.iotd_position)

        reward_success, reward_fail = 0.0, 0.0
        
        # --- MODIFIED: TDMA-based sequential interaction (Charge then Collect) ---
        for uav_idx, iot_idx in self.current_assignments:
            # Skip if the UAV doesn't have enough energy for the task
            if self.uav_energy_levels[uav_idx] < self.uav_comm_energy_cost:
                reward_fail -= 0.5 # Small penalty for UAV lacking energy
                continue

            collector_uav_pos = self.uav_positions[uav_idx]
            
            # --- Phase 1: Energy Transfer (UAV -> IoT) ---
            dist_UAV_IoTD = torch.linalg.norm(collector_uav_pos - self.iotd_position[iot_idx])
            energy_harvested = self.P_tx_UAV * (self.beta_0 / (dist_UAV_IoTD**2 + 1e-9)) * self.eta
            self.iot_energy_levels[iot_idx] += energy_harvested
            
            # --- Phase 2: Data Collection (IoT -> UAV) ---
            if self.iot_energy_levels[iot_idx] >= self.iot_comm_drain_rate:
                # IoT device consumes energy to send data
                self.iot_energy_levels[iot_idx] -= self.iot_comm_drain_rate

                secure_rate = self._calculate_secure_rate_torch(collector_uav_pos, self.iotd_position[iot_idx])
                
                if secure_rate > self.R_min:
                    reward_success += 10.0
                    self.AoI[iot_idx] = 0.0
                else:
                    reward_fail -= 1.0 # Failure due to low secure rate
            else:
                reward_fail -= 1.0 # Failure due to insufficient IoT energy
            
            # NEW: UAV consumes energy for performing the communication task
            self.uav_energy_levels[uav_idx] -= self.uav_comm_energy_cost
        
        self.iot_energy_levels.clamp_(min=0.0, max=1.0)
        self.uav_energy_levels.clamp_(min=0.0, max=1.0)
        
        aoi_penalty = -aoi_penalty_weight * torch.mean(self.AoI).item()
        energy_penalty = -energy_penalty_weight * propulsion_energy.item()
        
        reward = reward_success + reward_fail + aoi_penalty + energy_penalty + collision_penalty + boundary_penalty
        
        terminated = self.time >= self.T
        info = {
            'propulsion_energy': propulsion_energy.item(), 
            'sum_AoI': torch.sum(self.AoI).item(), 
            'collisions': self.collision_count,
            'avg_iot_energy': torch.mean(self.iot_energy_levels).item(),
            'avg_uav_energy': torch.mean(self.uav_energy_levels).item()
        }
        
        return self._get_obs(), reward, terminated, False, info

    def _check_boundaries_torch(self):
        """Checks if any UAV has flown out of the designated area."""
        lower, upper = torch.tensor([0.,0.,0.], device=self.device), self.size
        out_of_bounds = torch.any((self.uav_positions < lower) | (self.uav_positions > upper), dim=1)
        penalty = -1.0 * torch.sum(out_of_bounds).item()
        self.uav_positions.clamp_(min=0.)
        for i in range(3): self.uav_positions[:, i].clamp_(max=self.size[i])
        return penalty

    def _check_collisions_torch(self):
        """Checks for collisions between any pair of UAVs."""
        penalty = 0.0
        for uav1_idx, uav2_idx in combinations(range(self.num_uavs), 2):
            dist = torch.linalg.norm(self.uav_positions[uav1_idx] - self.uav_positions[uav2_idx])
            if dist < self.collision_threshold:
                penalty -= 1.0
                self.collision_count += 1
        return penalty

    def _calculate_secure_rate_torch(self, collector_pos, iotd_pos):
        """Calculates the secure communication rate from an IoT device to a collector."""
        jammer_pos = self.uav_positions[2] # Assumes the 3rd UAV is the jammer
        get_gain = lambda p1, p2: self.beta_0 / (torch.sum((p1-p2)**2) + 1e-9)
        
        sig_main = self.P_tx_IoTD * get_gain(iotd_pos, collector_pos)
        inter_main = self.P_jammer * get_gain(jammer_pos, collector_pos) + self.noise_power
        rate_main = torch.log2(1.0 + sig_main / inter_main)
        
        sig_eve = self.P_tx_IoTD * get_gain(iotd_pos, self.eavesdropper_pos)
        inter_eve = self.P_jammer * get_gain(jammer_pos, self.eavesdropper_pos) + self.noise_power
        rate_eve = torch.log2(1.0 + sig_eve / inter_eve)
        
        return torch.clamp(rate_main - rate_eve, min=0.0)


def plot_and_save_results(log_df, filename="masac_training_performance.png"):
    fig, axs = plt.subplots(6, 1, figsize=(12, 30), sharex=True)
    fig.suptitle('MASAC with LLM Guidance Training Performance', fontsize=18)

    axs[0].plot(log_df['episode'], log_df['reward'], color='green')
    axs[0].set_title("Episodic Reward"); axs[0].set_ylabel("Reward")
    axs[0].grid(True)
    axs[1].plot(log_df['episode'], log_df['avg_critic_loss'], color='red')
    axs[1].set_title("Average Critic Loss"); axs[1].set_ylabel("Loss")
    axs[1].grid(True)
    axs[2].plot(log_df['episode'], log_df['avg_actor_loss'], color='blue')
    axs[2].set_title("Average Actor Loss"); axs[2].set_ylabel("Loss")
    axs[2].grid(True)
    axs[3].plot(log_df['episode'], log_df['avg_propulsion_energy'], color='purple')
    axs[3].set_title("Average Propulsion Energy"); axs[3].set_ylabel("Energy")
    axs[3].grid(True)
    axs[4].plot(log_df['episode'], log_df['avg_sum_aoi'], color='orange')
    axs[4].set_title("Average Sum of AoI")
    axs[4].set_ylabel("AoI")
    axs[5].plot(log_df['episode'], log_df['total_collisions'], color='black')
    axs[5].set_title("Total Collisions per Episode")
    axs[5].set_ylabel("Collisions")
    axs[5].set_xlabel("Episode"); axs[5].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.savefig(filename); plt.close()
    print(f"\nPlots saved to {os.path.abspath(filename)}")

def main():
    env = MultiAgentEnv(num_uavs=3)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])

    agent = MASAC(
        global_state_dim=env.observation_space.shape[0],
        action_spaces=env.action_spaces,
        num_agents=env.num_agents,
        args=config
    )
    memory = ReplayBuffer(config['replay_size'])
    total_numsteps = 0
    training_logs = []

    for i_episode in range(config['max_episodes']):
        obs, _ = env.reset(seed=config['seed'] + i_episode)
        episode_reward, episode_steps, done = 0, 0, False
        
        episode_critic_losses, episode_actor_losses = [], []
        episode_propulsion_energy, episode_sum_aoi, episode_collisions = [], [], []

        while not done:
            if config['start_steps'] > total_numsteps:
                # Warm-up phase: purely random actions
                actions = np.array([space.sample() for space in env.action_spaces])
            else:
                if np.random.rand() < config['epsilon']:
                    # ε chance → exploration (random action for each agent)
                    actions = np.array([space.sample() for space in env.action_spaces])
                else:
                    # 1-ε chance → exploitation (policy action)
                    actions = agent.select_actions(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            global_state = env._get_global_state()
            next_global_state = env._get_global_state()
            memory.push(global_state, actions.flatten(), reward, next_global_state, done)
            if total_numsteps > config['batch_size']*5: 
                critic_loss, actor_losses = agent.update_parameters(memory)
                if critic_loss is not None:
                    episode_critic_losses.append(critic_loss)
                    episode_actor_losses.extend(actor_losses)

            episode_propulsion_energy.append(info['propulsion_energy'])
            episode_sum_aoi.append(info['sum_AoI'])
            
            obs = next_obs
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
        
        episode_collisions.append(info['collisions'])
        critic_loss_mean = np.mean(episode_critic_losses) if episode_critic_losses else -1
        actor_loss_mean = np.mean([v for v in episode_actor_losses if v is not None]) if episode_actor_losses else -1
        
        log_entry = {
            'episode': i_episode + 1, 'reward': episode_reward,
            'avg_critic_loss': critic_loss_mean, 'avg_actor_loss': actor_loss_mean,
            'avg_propulsion_energy': np.mean(episode_propulsion_energy),
            'avg_sum_aoi': np.mean(episode_sum_aoi),
            'total_collisions': np.sum(episode_collisions)
        }
        training_logs.append(log_entry)

        print(f"E: {i_episode+1}, R: {episode_reward/100:.2f}, Energy: {np.mean(episode_propulsion_energy):.2f}, "
              f"AoI: {np.mean(episode_sum_aoi):.2f}, Collisions: {np.sum(episode_collisions)}, "
              f"CL: {critic_loss_mean:.4f}, AL: {actor_loss_mean:.4f}")
        config['epsilon'] = max(config['epsilon'] * config['epsilon_decay'], config['epsilon_min'])


    env.close()

    log_df = pd.DataFrame(training_logs)
    csv_filename = 'masac_llm_training_logs.csv'
    log_df.to_csv(csv_filename, index=False)
    print(f"\nTraining logs saved to {os.path.abspath(csv_filename)}")
    plot_and_save_results(log_df)

if __name__ == '__main__':
    main()
