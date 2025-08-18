import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.spaces import Box
import random

# --- Part 0: GPU/CPU Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Part 1: Environment Definition (Slightly Refined) ---
class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        # --- Environment Parameters ---
        self.size = np.array([300, 300, 300])
        self.num_IoTD = 5
        self.T = 150

        # --- UAV & IoTD Initial State ---
        self.start_position_A = np.array([0, 299, 0], dtype=float)
        self.start_position_B = np.array([299, 299, 200], dtype=float)
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ], dtype=float)
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ], dtype=float)

        # --- Physics & Communication Parameters ---
        self.R_min = 0.1
        self.P_tx_UAV = 0.5
        self.P_circuit_UAV = 0.05
        self.eta = 0.5
        self.beta_0 = 1e-3
        self.noise_power = 1e-13
        self.IoTD_idle_drain = 0.001
        self.IoTD_comm_drain = 0.1

        # --- Action & Observation Spaces ---
        action_dim = 6 + 1 + self.num_IoTD # UAV_A(3), UAV_B(3), rho(1), delta(5)
        low_action = np.array([-1.0] * 6 + [0.0] * (1 + self.num_IoTD))
        high_action = np.array([1.0] * action_dim)
        self.action_space = Box(low=low_action, high=high_action, dtype=np.float32)
        
        # State variables are initialized in reset()
        self.reset()

    def reset(self):
        self.current_position_A = self.start_position_A.copy()
        self.current_position_B = self.start_position_B.copy()
        self.AoI = np.zeros(self.num_IoTD)
        self.energy_levels = np.ones(self.num_IoTD)
        self.time = 0
        self.E = 0
        self.A = 0
        self.done = False
        return build_graph_from_env(self)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx_A, dy_A, dz_A = action[0:3]
        dx_B, dy_B, dz_B = action[3:6]
        rho = action[6]
        delta = action[7:]

        r_A, r_E, r_P = 0, 0, 0
        self.time += 1
        if self.time >= self.T:
            self.done = True
            return build_graph_from_env(self), 0, self.done, {"energy":0,'aoi':0}

        # Movement and Propulsion Energy
        past_position_A = self.current_position_A.copy()
        past_position_B = self.current_position_B.copy()
        self.current_position_A += np.array([dx_A, dy_A, dz_A]) * 20
        self.current_position_B += np.array([dx_B, dy_B, dz_B]) * 20

        # Boundary checks
        if not np.all((self.current_position_A >= 0) & (self.current_position_A <= self.size)):
            self.current_position_A = np.clip(self.current_position_A, 0, self.size)
            r_P -= 10
        if not np.all((self.current_position_B >= 0) & (self.current_position_B <= self.size)):
            self.current_position_B = np.clip(self.current_position_B, 0, self.size)
            r_P -= 10

        propulsion_energy = self._calculate_propulsion_energy(past_position_A, self.current_position_A) + \
                            self._calculate_propulsion_energy(past_position_B, self.current_position_B)
        self.E += propulsion_energy

        # IoTD Idle Drain & AoI Update
        self.energy_levels = np.maximum(0, self.energy_levels - self.IoTD_idle_drain)
        self.AoI += 1

        # Communication Phase
        selected_iotd = np.argmax(delta)
        time_comm = rho
        time_energy_transfer = time_comm * 0.5
        time_data_collection = time_comm * 0.5

        # Energy Transfer
        dist_UAV_IoTD = np.linalg.norm(self.current_position_A - self.iotd_position[selected_iotd])
        channel_gain = self.beta_0 / (dist_UAV_IoTD**2 + 1e-9)
        energy_harvested = self.P_tx_UAV * channel_gain * time_energy_transfer * self.eta
        if self.energy_levels[selected_iotd] < 0.99:
            r_P += 5 * energy_harvested * 100
        self.energy_levels[selected_iotd] = min(1.0, self.energy_levels[selected_iotd] + energy_harvested)
        uav_tx_energy = self.P_tx_UAV * time_energy_transfer
        self.E += uav_tx_energy

        # Data Collection
        secure_rates = self._calculate_secure_rates()
        uav_rx_energy = 0
        if self.energy_levels[selected_iotd] > self.IoTD_comm_drain and secure_rates[selected_iotd] > self.R_min:
            r_P += 10
            self.A += self.AoI[selected_iotd]
            r_A = -0.1 * self.AoI[selected_iotd]
            self.AoI[selected_iotd] = 0
            self.energy_levels[selected_iotd] -= self.IoTD_comm_drain
            uav_rx_energy = self.P_circuit_UAV * time_data_collection
            self.E += uav_rx_energy
        elif rho > 0.1:
            r_P -= 5

        # Final Reward
        r_E = -0.01 * (propulsion_energy + uav_tx_energy + uav_rx_energy)
        reward = r_A + r_E + r_P
        return build_graph_from_env(self), reward, self.done, {"energy":self.E,'aoi':self.A}

    def _calculate_propulsion_energy(self, pos_start, pos_end):
        distance = np.linalg.norm(pos_end - pos_start)
        return 0.05 * distance**2

    def _calculate_secure_rates(self):
        p_iotd_tx = 0.1
        p_jam = 0.1
        dist_IoTD_to_A = np.linalg.norm(self.iotd_position - self.current_position_A, axis=1)
        dist_B_to_A = np.linalg.norm(self.current_position_A - self.current_position_B)
        h_IoTD_A = self.beta_0 / (dist_IoTD_to_A**2 + 1e-9)
        h_B_A = self.beta_0 / (dist_B_to_A**2 + 1e-9)
        rate_at_uav = np.log2(1 + (p_iotd_tx * h_IoTD_A) / (p_jam * h_B_A + self.noise_power))
        max_rate_at_eavesdropper = np.zeros(self.num_IoTD)
        for i in range(self.num_IoTD):
            dist_IoTD_e = np.linalg.norm(self.e_position - self.iotd_position[i], axis=1)
            dist_B_e = np.linalg.norm(self.e_position - self.current_position_B, axis=1)
            h_IoTD_e = self.beta_0 / (dist_IoTD_e**2 + 1e-9)
            h_B_e = self.beta_0 / (dist_B_e**2 + 1e-9)
            rate_at_e = np.log2(1 + (p_iotd_tx * h_IoTD_e) / (p_jam * h_B_e + self.noise_power))
            max_rate_at_eavesdropper[i] = np.max(rate_at_e)
        return np.maximum(0, rate_at_uav - max_rate_at_eavesdropper)

# --- Part 2: Graph Construction ---
def build_graph_from_env(env, distance_threshold=200.0):
    num_uavs, num_iotds, num_eaves = 2, env.num_IoTD, len(env.e_position)
    num_nodes = num_uavs + num_iotds + num_eaves
    n_features = 8

    node_features = np.zeros((num_nodes, n_features), dtype=np.float32)
    positions = np.zeros((num_nodes, 3), dtype=np.float32)

    # Fill features and positions
    node_features[0, :] = np.concatenate([env.current_position_A / env.size, [1, 0, 0, 0, 0]])
    positions[0, :] = env.current_position_A
    node_features[1, :] = np.concatenate([env.current_position_B / env.size, [1, 0, 0, 0, 0]])
    positions[1, :] = env.current_position_B
    for i in range(num_iotds):
        idx = num_uavs + i
        node_features[idx, :] = np.concatenate([env.iotd_position[i] / env.size, [0, 1, 0, env.AoI[i] / env.T, env.energy_levels[i]]])
        positions[idx, :] = env.iotd_position[i]
    for i in range(num_eaves):
        idx = num_uavs + num_iotds + i
        node_features[idx, :] = np.concatenate([env.e_position[i] / env.size, [0, 0, 1, 0, 0]])
        positions[idx, :] = env.e_position[i]

    # Build Adjacency Matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if i == j:
                adj_matrix[i, j] = 1.0
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < distance_threshold:
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
    return node_features, adj_matrix

# --- Part 3: PyTorch GNN, Actor, Critic, and Replay Buffer ---
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, adj_matrix):
        # Symmetrically normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=-1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        norm_adj = torch.bmm(degree_matrix, torch.bmm(adj_matrix, degree_matrix))
        
        # Propagate features
        transformed_features = self.linear(node_features)
        output = torch.bmm(norm_adj, transformed_features)
        return F.relu(output)

class GNNBase(nn.Module):
    def __init__(self, n_node_features):
        super(GNNBase, self).__init__()
        self.gcn1 = GCNLayer(n_node_features, 128)
        self.gcn2 = GCNLayer(128, 64)

    def forward(self, node_features, adj_matrix):
        x = self.gcn1(node_features, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        # Extract UAV embeddings (nodes 0 and 1)
        uav_a_embedding = x[:, 0, :]
        uav_b_embedding = x[:, 1, :]
        return torch.cat([uav_a_embedding, uav_b_embedding], dim=1)

class Critic(nn.Module):
    def __init__(self, n_node_features, action_dim):
        super(Critic, self).__init__()
        self.gnn_base = GNNBase(n_node_features)
        self.q_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, node_features, adj_matrix, action):
        gnn_output = self.gnn_base(node_features, adj_matrix)
        state_action = torch.cat([gnn_output, action], dim=1)
        return self.q_net(state_action)

class Actor(nn.Module):
    def __init__(self, n_node_features, action_dim):
        super(Actor, self).__init__()
        self.gnn_base = GNNBase(n_node_features)
        self.policy_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, node_features, adj_matrix):
        gnn_output = self.gnn_base(node_features, adj_matrix)
        return self.policy_net(gnn_output)

class ReplayBuffer:
    def __init__(self, max_size, node_shape, action_dim):
        self.max_size = int(max_size)
        self.mem_cntr = 0
        self.node_features_memory = np.zeros((self.max_size, *node_shape), dtype=np.float32)
        self.adj_matrix_memory = np.zeros((self.max_size, node_shape[0], node_shape[0]), dtype=np.float32)
        self.action_memory = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.next_node_features_memory = np.zeros((self.max_size, *node_shape), dtype=np.float32)
        self.next_adj_matrix_memory = np.zeros((self.max_size, node_shape[0], node_shape[0]), dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.max_size
        self.node_features_memory[index], self.adj_matrix_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_node_features_memory[index], self.next_adj_matrix_memory[index] = next_state
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            torch.tensor(self.node_features_memory[batch], device=device),
            torch.tensor(self.adj_matrix_memory[batch], device=device),
            torch.tensor(self.action_memory[batch], device=device),
            torch.tensor(self.reward_memory[batch], device=device),
            torch.tensor(self.next_node_features_memory[batch], device=device),
            torch.tensor(self.next_adj_matrix_memory[batch], device=device),
            torch.tensor(self.terminal_memory[batch], device=device)
        )

# --- Part 4: TD3 Agent in PyTorch ---
class TD3Agent:
    def __init__(self, env, n_nodes, n_node_features, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.env = env
        self.n_nodes, self.n_node_features = n_nodes, n_node_features
        self.action_dim = env.action_space.shape[0]
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay

        self.actor = Actor(n_node_features, self.action_dim).to(device)
        self.critic_1 = Critic(n_node_features, self.action_dim).to(device)
        self.critic_2 = Critic(n_node_features, self.action_dim).to(device)
        self.target_actor = Actor(n_node_features, self.action_dim).to(device)
        self.target_critic_1 = Critic(n_node_features, self.action_dim).to(device)
        self.target_critic_2 = Critic(n_node_features, self.action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(1e6, (n_nodes, n_node_features), self.action_dim)
        self.learn_step_counter = 0

    def choose_action(self, state, evaluate=False):
        self.actor.eval()
        node_features, adj_matrix = state
        node_features = torch.tensor([node_features], dtype=torch.float32).to(device)
        adj_matrix = torch.tensor([adj_matrix], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            mu = self.actor(node_features, adj_matrix)
        
        if not evaluate:
            noise = torch.normal(mean=0.0, std=0.1, size=(self.action_dim,)).to(device)
            mu = mu + noise
        
        self.actor.train()
        action = torch.clamp(mu, torch.tensor(self.env.action_space.low, device=device), 
                             torch.tensor(self.env.action_space.high, device=device))
        return action.cpu().numpy().flatten()

    def learn(self, batch_size=256):
        if self.memory.mem_cntr < batch_size:
            return None, None

        node_feats, adj_mats, actions, rewards, next_node_feats, next_adj_mats, dones = self.memory.sample_buffer(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_node_feats, next_adj_mats) + noise).clamp(
                torch.tensor(self.env.action_space.low, device=device), 
                torch.tensor(self.env.action_space.high, device=device)
            )
            q1_next = self.target_critic_1(next_node_feats, next_adj_mats, next_actions)
            q2_next = self.target_critic_2(next_node_feats, next_adj_mats, next_actions)
            q_next = torch.min(q1_next, q2_next).view(-1)
            target_q = rewards + self.gamma * (1 - dones.float()) * q_next

        # --- Critic Update ---
        q1 = self.critic_1(node_feats, adj_mats, actions).view(-1)
        q2 = self.critic_2(node_feats, adj_mats, actions).view(-1)
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0) # Gradient Clipping
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0) # Gradient Clipping
        self.critic_2_optimizer.step()

        critic_loss = critic_1_loss + critic_2_loss
        actor_loss = None

        # --- Delayed Actor Update ---
        if self.learn_step_counter % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            new_actions = self.actor(node_feats, adj_mats)
            q_for_actor = self.critic_1(node_feats, adj_mats, new_actions)
            actor_loss = -torch.mean(q_for_actor)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.update_target_networks()
        
        self.learn_step_counter += 1
        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# --- Part 5: Main Training Loop ---
def main():
    env = Environment()
    sample_state = env.reset()
    n_nodes, n_node_features = sample_state[0].shape
    
    agent = TD3Agent(env, n_nodes, n_node_features)
    
    total_episodes = 1000
    batch_size = 256
    start_training_steps = 2000
    total_steps = 0
    
    reward_history, avg_actor_loss_hist, avg_critic_loss_hist , energy_hist,aoi_hist = [], [], [],[],[]

    print(f"Starting training for {total_episodes} episodes...")
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        ep_reward, ep_steps ,ep_energy,ep_aoi = 0, 0,0,0
        ep_actor_losses, ep_critic_losses = [], []

        while not done:
            if total_steps < start_training_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
            
            next_state, reward, done, info_dict = env.step(action)
            agent.memory.store_transition(state, action, reward, next_state, done)
            ep_energy+=info_dict['energy']
            ep_aoi+=info_dict['aoi']
            
            if total_steps >= start_training_steps:
                c_loss, a_loss = agent.learn(batch_size)
                if c_loss is not None: ep_critic_losses.append(c_loss)
                if a_loss is not None: ep_actor_losses.append(a_loss)

            ep_reward += reward
            state = next_state
            total_steps += 1
            ep_steps += 1
        
        reward_history.append(ep_reward/150)
        energy_hist.append(ep_energy/150)
        aoi_hist.append(ep_aoi/150)
        avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0
        avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0
        avg_critic_loss_hist.append(avg_critic_loss/150)
        avg_actor_loss_hist.append(avg_actor_loss/150)

        print(f"Ep: {episode+1:4} | Reward: {ep_reward/150:8.2f} | C-Loss: {avg_critic_loss/150:8.4f} | A-Loss: {avg_actor_loss/150:8.4f} | Energy: {ep_energy/150:8.2f} | AoI: {ep_aoi/150:8.2f} |")
    
        print("------------------------------------------------------------------------")
    # Save results
    results_df = pd.DataFrame({
        'Episode': range(1, total_episodes + 1),
        'Total Reward': reward_history,
        'Average Critic Loss': avg_critic_loss_hist,
        'Average Actor Loss': avg_actor_loss_hist,
        'Avg Energy': energy_hist,
        'aoi':aoi_hist
    })
    results_df.to_csv("training_log_gnn_test_01.csv", index=False)
    print("\nTraining Complete. Results saved to 'training_log_gnn_test_01.csv'")

if __name__ == "__main__":
    main()
