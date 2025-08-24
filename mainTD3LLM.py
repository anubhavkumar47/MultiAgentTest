import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pandas as pd
import random
from gym.spaces import Box
from itertools import combinations

# --- Part 0: GPU/CPU Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Part 1: Environment Definition (State representation MODIFIED) ---
class Environment:
    """
    Defines the multi-agent simulation environment for UAVs and IoT devices.
    It manages state, actions, rewards, and dynamics, including collision avoidance
    and simulated LLM guidance.
    """
    def __init__(self, num_uavs=3):
        super(Environment, self).__init__()
        self.num_uavs = num_uavs
        # --- Environment Parameters ---
        self.size = np.array([300, 300, 300])
        self.num_IoTD = 5
        self.T = 150 # Max timesteps per episode

        # --- UAV & IoTD Initial State ---
        self.start_positions = np.array([
            [0, 299, 100],
            [150, 299, 100],
            [299, 299, 100]
        ], dtype=float)
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ], dtype=float)
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ], dtype=float)

        # --- Physics, Communication & Safety Parameters ---
        self.R_min = 0.1
        self.P_tx_UAV = 0.5
        self.P_circuit_UAV = 0.05
        self.eta = 0.5
        self.beta_0 = 1e-3
        self.noise_power = 1e-13
        self.IoTD_idle_drain = 0.001
        self.IoTD_comm_drain = 0.1
        self.collision_threshold = 25.0 # Min safe distance between UAVs

        # --- Action & Observation Spaces ---
        action_dim = (self.num_uavs * 3) + 1 + self.num_IoTD
        low_action = np.array([-1.0] * (self.num_uavs * 3) + [0.0] * (1 + self.num_IoTD))
        high_action = np.array([1.0] * action_dim)
        self.action_space = Box(low=low_action, high=high_action, dtype=np.float32)

        self.reset()

    def reset(self):
        """Resets the environment to its initial state."""
        self.uav_positions = self.start_positions[:self.num_uavs].copy()
        self.AoI = np.zeros(self.num_IoTD)
        self.energy_levels = np.ones(self.num_IoTD)
        self.time = 0
        self.E = 0 # Total energy consumption
        self.A = 0 # Total AoI collected
        self.done = False
        return self._get_state()

    # --- MODIFIED: State representation is now a flat vector ---
    def _get_state(self):
        """
        Constructs a flat state vector from the environment's properties.
        This replaces the graph-based state representation.
        """
        # Normalize positions by the environment size
        uav_pos_norm = self.uav_positions.flatten() / self.size[0]
        iotd_pos_norm = self.iotd_position.flatten() / self.size[0]
        
        # Normalize AoI and time by the max episode length
        aoi_norm = self.AoI / self.T
        time_norm = np.array([self.time / self.T])
        
        # Energy levels are already in [0, 1]
        
        # Concatenate all features into a single flat vector
        state = np.concatenate([
            uav_pos_norm,
            iotd_pos_norm,
            aoi_norm,
            self.energy_levels,
            time_norm
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        """Executes one time step in the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        uav_actions = action[:self.num_uavs * 3].reshape(self.num_uavs, 3)
        rho = action[self.num_uavs * 3]
        delta = action[self.num_uavs * 3 + 1:]

        r_A, r_E, r_P = 0, 0, 0
        self.time += 1
        if self.time >= self.T:
            self.done = True
            return self._get_state(), 0, self.done, {"energy": self.E, 'aoi': self.A}

        past_positions = self.uav_positions.copy()
        self.uav_positions += uav_actions * 20

        collision_penalty = 0
        for i in range(self.num_uavs):
            if not np.all((self.uav_positions[i] >= 0) & (self.uav_positions[i] <= self.size)):
                self.uav_positions[i] = np.clip(self.uav_positions[i], 0, self.size)
                r_P -= 10

        for uav1_idx, uav2_idx in combinations(range(self.num_uavs), 2):
            dist = np.linalg.norm(self.uav_positions[uav1_idx] - self.uav_positions[uav2_idx])
            if dist < self.collision_threshold:
                collision_penalty -= 50

        propulsion_energy = np.sum([self._calculate_propulsion_energy(past_positions[i], self.uav_positions[i]) for i in range(self.num_uavs)])
        self.E += propulsion_energy

        self.energy_levels = np.maximum(0, self.energy_levels - self.IoTD_idle_drain)
        self.AoI += 1

        primary_uav_pos = self.uav_positions[0]
        selected_iotd = np.argmax(delta)

        dist_UAV_IoTD = np.linalg.norm(primary_uav_pos - self.iotd_position[selected_iotd])
        channel_gain = self.beta_0 / (dist_UAV_IoTD**2 + 1e-9)
        energy_harvested = self.P_tx_UAV * channel_gain * rho * 0.5 * self.eta
        self.energy_levels[selected_iotd] = min(1.0, self.energy_levels[selected_iotd] + energy_harvested)
        uav_tx_energy = self.P_tx_UAV * rho * 0.5
        self.E += uav_tx_energy

        secure_rates = self._calculate_secure_rates(primary_uav_pos)
        uav_rx_energy = 0
        if self.energy_levels[selected_iotd] > self.IoTD_comm_drain and secure_rates[selected_iotd] > self.R_min:
            r_P += 10
            self.A += self.AoI[selected_iotd]
            r_A = -0.1 * self.AoI[selected_iotd]
            self.AoI[selected_iotd] = 0
            self.energy_levels[selected_iotd] -= self.IoTD_comm_drain
            uav_rx_energy = self.P_circuit_UAV * rho * 0.5
            self.E += uav_rx_energy
        elif rho > 0.1:
            r_P -= 5

        r_E = -0.01 * (propulsion_energy + uav_tx_energy + uav_rx_energy)
        reward = r_A + r_E + r_P + collision_penalty
        return self._get_state(), reward, self.done, {"energy": self.E, 'aoi': self.A}

    def _calculate_propulsion_energy(self, pos_start, pos_end):
        return 0.1 * np.linalg.norm(pos_end - pos_start)

    def _calculate_secure_rates(self, primary_uav_pos):
        jamming_uav_pos = self.uav_positions[-1]
        p_iotd_tx, p_jam = 0.1, 0.1
        dist_IoTD_to_A = np.linalg.norm(self.iotd_position - primary_uav_pos, axis=1)
        dist_B_to_A = np.linalg.norm(primary_uav_pos - jamming_uav_pos)
        h_IoTD_A = self.beta_0 / (dist_IoTD_to_A**2 + 1e-9)
        h_B_A = self.beta_0 / (dist_B_to_A**2 + 1e-9)
        return np.log2(1 + (p_iotd_tx * h_IoTD_A) / (p_jam * h_B_A + self.noise_power))

# --- Part 2: Graph Construction ---
# --- REMOVED: build_graph_from_env, GCNLayer, and GNNBase are no longer needed. ---

# --- Part 3: Neural Network Models (MODIFIED to use MLPs) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        if not isinstance(max_action, torch.Tensor):
             max_action = torch.tensor(max_action, device=device, dtype=torch.float32)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# --- MODIFIED: Replay buffer for flat states ---
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.max_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            torch.tensor(self.state_memory[batch], device=device),
            torch.tensor(self.action_memory[batch], device=device),
            torch.tensor(self.reward_memory[batch], device=device),
            torch.tensor(self.next_state_memory[batch], device=device),
            torch.tensor(self.terminal_memory[batch], device=device)
        )

# --- Part 4: TD3 Agent (MODIFIED to use new networks and buffer) ---
class TD3Agent:
    def __init__(self, env, state_dim, action_dim, lr_actor=1e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 expl_noise=0.1, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.env = env
        self.action_dim = action_dim
        self.max_action_val = env.action_space.high[0] # For scaling noise
        
        self.max_action = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.min_action = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
        self.expl_noise = expl_noise
        self.epsilon, self.epsilon_end, self.epsilon_decay = epsilon_start, epsilon_end, epsilon_decay

        self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(1e5, state_dim, action_dim)

    def choose_action(self, state, add_noise=True):
        if add_noise and random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = torch.tensor([state], dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self.actor(state)

        if add_noise:
            noise = (torch.randn_like(action) * self.expl_noise).to(device)
            action = action + noise

        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().cpu().numpy().flatten()

    def learn(self, batch_size=256):
        if self.memory.mem_cntr < batch_size: return
        self.total_it += 1

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)
        rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, self.min_action, self.max_action)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones.float()) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

# --- Part 5: Main Training Loop (MODIFIED for new state_dim) ---
def main():
    env = Environment(num_uavs=3)
    
    # --- MODIFIED: Get state and action dimensions from the env ---
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TD3Agent(env, state_dim, action_dim)

    total_episodes = 10000
    batch_size = 256
    start_training_steps = 1000
    total_steps = 0

    reward_history, energy_hist, aoi_hist, epsilon_hist = [], [], [], []

    print(f"Starting TD3 training for {total_episodes} episodes...")
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        ep_reward, ep_steps, ep_energy, ep_aoi = 0, 0, 0, 0

        while not done:
            if total_steps < start_training_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state, add_noise=True)

            next_state, reward, done, info_dict = env.step(action)
            agent.memory.store_transition(state, action, reward, next_state, done)

            if 'energy' in info_dict: ep_energy = info_dict['energy']
            if 'aoi' in info_dict: ep_aoi = info_dict['aoi']

            if total_steps >= start_training_steps:
                agent.learn(batch_size)

            ep_reward += reward
            state = next_state
            total_steps += 1
            ep_steps += 1
        
        avg_reward = ep_reward / ep_steps if ep_steps > 0 else 0
        reward_history.append(avg_reward)
        energy_hist.append(ep_energy)
        aoi_hist.append(ep_aoi)
        epsilon_hist.append(agent.epsilon)
        
        print(f"Ep: {episode+1:4} | Avg Reward: {avg_reward:8.2f} | Total Energy: {ep_energy:8.2f} | Total AoI: {ep_aoi:8.2f} | Epsilon: {agent.epsilon:.4f}")

    print("\nTraining Complete.")
    results_df = pd.DataFrame({
        'Episode': range(1, total_episodes + 1), 
        'Average Reward': reward_history,
        'Total Energy': energy_hist,
        'Total AoI': aoi_hist,
        'Epsilon': epsilon_hist
    })
    results_df.to_csv("training_log_multi_agent_td3_mlp.csv", index=False)
    print("Results saved to 'training_log_multi_agent_td3_mlp.csv'")

if __name__ == '__main__':
    main()
