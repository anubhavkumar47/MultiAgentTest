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
    """
    def __init__(self, num_uavs=3):
        super(Environment, self).__init__()
        self.num_uavs = num_uavs
        self.size = np.array([300, 300, 300])
        self.num_IoTD = 5
        self.T = 150

        self.start_positions = np.array([
            [0, 299, 100], [150, 299, 100], [299, 299, 100]
        ], dtype=float)
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ], dtype=float)
        
        self.R_min = 0.1
        self.P_tx_UAV = 0.5
        self.P_circuit_UAV = 0.05
        self.eta = 0.5
        self.beta_0 = 1e-3
        self.noise_power = 1e-13
        self.IoTD_idle_drain = 0.001
        self.IoTD_comm_drain = 0.1
        self.collision_threshold = 25.0

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
        self.E = 0
        self.A = 0
        self.done = False
        return self._get_state()

    # --- MODIFIED: State representation is now a flat vector ---
    def _get_state(self):
        """
        Constructs a flat state vector from the environment's properties.
        This replaces the graph-based state representation.
        """
        uav_pos_norm = self.uav_positions.flatten() / self.size[0]
        iotd_pos_norm = self.iotd_position.flatten() / self.size[0]
        aoi_norm = self.AoI / self.T
        time_norm = np.array([self.time / self.T])
        
        state = np.concatenate([
            uav_pos_norm, iotd_pos_norm, aoi_norm, self.energy_levels, time_norm
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

# --- REMOVED: All GNN-related classes and functions are removed. ---

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = torch.tensor(max_action, device=device, dtype=torch.float32)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean_layer(a)
        log_std = self.log_std_layer(a).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

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

# --- Part 4: SAC Agent (MODIFIED for MLP architecture) ---
class SACAgent:
    def __init__(self, env, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005):
        self.env = env
        self.action_dim = action_dim
        self.max_action = env.action_space.high
        self.gamma, self.tau = gamma, tau

        self.actor = Actor(state_dim, self.action_dim, self.max_action).to(device)
        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = Critic(state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        self.alpha = self.log_alpha.exp().item()

        self.memory = ReplayBuffer(1e5, state_dim, self.action_dim)

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(device)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def learn(self, batch_size=256):
        if self.memory.mem_cntr < batch_size: return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)
        rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones.float()) * (torch.min(q1_next, q2_next) - self.alpha * next_log_pi)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        actor_loss = ((self.alpha * log_pi) - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        self.update_target_networks()
        return actor_loss.item(), critic_loss.item()

    def update_target_networks(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# --- Part 5: Main Training Loop (MODIFIED for new state_dim) ---
def main():
    env = Environment(num_uavs=3)

    # --- MODIFIED: Get state and action dimensions from the env ---
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(env, state_dim, action_dim)

    total_episodes = 10
    batch_size = 256
    start_training_steps = 1000
    total_steps = 0

    reward_history, energy_hist, aoi_hist = [], [], []
    closs,aloss =[],[]

    print(f"Starting training for {total_episodes} episodes...")
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        ep_reward, ep_steps, ep_energy, ep_aoi = 0, 0, 0, 0
        ep_closs,ep_aloss =0,0

        while not done:
            action = env.action_space.sample() if total_steps < start_training_steps else agent.choose_action(state)

            next_state, reward, done, info_dict = env.step(action)
            agent.memory.store_transition(state, action, reward, next_state, done)

            if 'energy' in info_dict: ep_energy = info_dict['energy']
            if 'aoi' in info_dict: ep_aoi = info_dict['aoi']

            if total_steps >= start_training_steps:
                actor_loss,critic_loss=agent.learn(batch_size)
                ep_closs += critic_loss
                ep_aloss+=actor_loss

            ep_reward += reward
            state = next_state
            total_steps += 1
            ep_steps += 1

        avg_reward = ep_reward / ep_steps if ep_steps > 0 else 0
        final_energy = ep_energy
        final_aoi = ep_aoi
        
        reward_history.append(avg_reward)
        energy_hist.append(final_energy)
        aloss.append(ep_aloss)
        closs.append(ep_closs)
        aoi_hist.append(final_aoi)
        
        print(f"Ep: {episode+1:4} | Avg Reward: {avg_reward:8.2f} | Total Energy: {final_energy:8.2f} | Total AoI: {final_aoi:8.2f}")

    print("\nTraining Complete.")
    results_df = pd.DataFrame({
        'Episode': range(1, total_episodes + 1), 
        'Average Reward': reward_history,
        'Total Energy': energy_hist,
        'Total AoI': aoi_hist,
        'Actor Loss': aloss,
        'Critic Loss': closs ,
        })
    results_df.to_csv("training_log_multi_agent_sac_mlp.csv", index=False)
    print("Results saved to 'training_log_multi_agent_sac_mlp.csv'")

if __name__ == '__main__':
    main()
