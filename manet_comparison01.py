import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj

# --- Hyperparameters ---
NUM_NODES = 10
EMBED_DIM = 64
GNN_LAYERS = 3
DQN_HIDDEN = 256
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.99
REPLAY_SIZE = 100_000
EPISODES = 1000
MAX_TTL = 20
EPSILON_MIN = 0.01
EPSILON_DECAY = 100

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- GNN Encoder ---
class GNNEncoder(nn.Module):
    def __init__(self, node_feat_dim, embed_dim=EMBED_DIM, layers=GNN_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed_dim = embed_dim
        for l in range(layers):
            in_dim = node_feat_dim if l == 0 else embed_dim
            self.layers.append(nn.Linear(in_dim + node_feat_dim, embed_dim))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x: [num_nodes, node_feat_dim]
        h = x
        for layer in self.layers:
            # Message passing: mean of neighbors' embeddings
            adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
            neighbor_mean = (adj @ h) / deg
            h = torch.cat([neighbor_mean, x], dim=1)
            h = self.relu(layer(h))
        return h  # [num_nodes, embed_dim]

# --- DQN Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, embed_dim, dest_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + dest_dim, DQN_HIDDEN)
        self.fc2 = nn.Linear(DQN_HIDDEN, DQN_HIDDEN)
        self.fc3 = nn.Linear(DQN_HIDDEN, action_dim)
        self.relu = nn.ReLU()

    def forward(self, node_embed, dest_feat):
        # node_embed: [embed_dim], dest_feat: [dest_dim]
        x = torch.cat([node_embed, dest_feat], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # [action_dim]

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Simple MANET Environment ---
class SimpleMANETEnv:
    def __init__(self, num_nodes=NUM_NODES):
        self.num_nodes = num_nodes
        self.positions = np.random.rand(num_nodes, 2) * 100  # 2D positions
        self.queues = np.zeros(num_nodes, dtype=int)
        self.link_quality = np.ones((num_nodes, num_nodes))
        self.adj_matrix = (np.linalg.norm(self.positions[:, None] - self.positions[None, :], axis=-1) < 40).astype(int)
        np.fill_diagonal(self.adj_matrix, 0)
        self.reset()

    def reset(self):
        self.queues[:] = 0
        self.packet_src = random.randint(0, self.num_nodes - 1)
        self.packet_dst = random.randint(0, self.num_nodes - 1)
        while self.packet_dst == self.packet_src:
            self.packet_dst = random.randint(0, self.num_nodes - 1)
        self.packet_pos = self.packet_src
        self.ttl = MAX_TTL
        return self._get_state()

    def _get_state(self):
        # Node features: [x, y, queue, link qualities to neighbors]
        node_feats = []
        for i in range(self.num_nodes):
            pos = self.positions[i]
            queue = [self.queues[i]]
            links = self.link_quality[i, :]
            node_feats.append(np.concatenate([pos, queue, links]))
        node_feats = np.stack(node_feats)
        edge_index = np.array(np.nonzero(self.adj_matrix))
        dest_feat = np.concatenate([self.positions[self.packet_dst], [self.packet_dst]])
        return node_feats, edge_index, dest_feat, self.packet_pos

    def step(self, action):
        # Action: next hop node index
        if self.adj_matrix[self.packet_pos, action] == 0:
            # Invalid action (not a neighbor)
            reward = -2.0
            done = True
            return self._get_state(), reward, done
        self.packet_pos = action
        self.queues[action] += 1
        self.ttl -= 1
        if self.packet_pos == self.packet_dst:
            reward = 10.0
            done = True
        elif self.ttl <= 0:
            reward = -5.0
            done = True
        else:
            reward = -0.1
            done = False
        return self._get_state(), reward, done

    def get_valid_actions(self, node_idx):
        return np.nonzero(self.adj_matrix[node_idx])[0]

# --- MA-DQN Agent ---
class MADQNAgent:
    def __init__(self, node_feat_dim, dest_dim, action_dim):
        self.gnn = GNNEncoder(node_feat_dim)
        self.qnet = QNetwork(EMBED_DIM, dest_dim, action_dim)
        self.target_qnet = QNetwork(EMBED_DIM, dest_dim, action_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(list(self.gnn.parameters()) + list(self.qnet.parameters()), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.tau = 0.001

    def select_action(self, node_feats, edge_index, dest_feat, node_idx, valid_actions, epsilon):
        self.gnn.eval()
        self.qnet.eval()
        with torch.no_grad():
            x = torch.tensor(node_feats, dtype=torch.float32)
            ei = torch.tensor(edge_index, dtype=torch.long)
            dest = torch.tensor(dest_feat, dtype=torch.float32)
            h = self.gnn(x, ei)
            node_embed = h[node_idx]
            q_values = self.qnet(node_embed, dest)
            q_values = q_values.cpu().numpy()
            # Mask invalid actions
            mask = np.full(q_values.shape, -np.inf)
            mask[valid_actions] = q_values[valid_actions]
            if random.random() < epsilon:
                return np.random.choice(valid_actions)
            else:
                return int(np.argmax(mask))

    def update(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return 0.0
        batch = replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*batch))
        # Prepare batch tensors
        node_feats_batch = []
        edge_index_batch = []
        dest_feat_batch = []
        node_idx_batch = []
        actions = []
        rewards = []
        next_node_feats_batch = []
        next_edge_index_batch = []
        next_dest_feat_batch = []
        next_node_idx_batch = []
        dones = []
        for i in range(BATCH_SIZE):
            nf, ei, df, ni = batch.state[i]
            node_feats_batch.append(torch.tensor(nf, dtype=torch.float32))
            edge_index_batch.append(torch.tensor(ei, dtype=torch.long))
            dest_feat_batch.append(torch.tensor(df, dtype=torch.float32))
            node_idx_batch.append(ni)
            actions.append(batch.action[i])
            rewards.append(batch.reward[i])
            nf2, ei2, df2, ni2 = batch.next_state[i]
            next_node_feats_batch.append(torch.tensor(nf2, dtype=torch.float32))
            next_edge_index_batch.append(torch.tensor(ei2, dtype=torch.long))
            next_dest_feat_batch.append(torch.tensor(df2, dtype=torch.float32))
            next_node_idx_batch.append(ni2)
            dones.append(batch.done[i])
        # Compute Q(s,a)
        q_values = []
        next_q_values = []
        for i in range(BATCH_SIZE):
            h = self.gnn(node_feats_batch[i], edge_index_batch[i])
            node_embed = h[node_idx_batch[i]]
            q = self.qnet(node_embed, dest_feat_batch[i])
            q_values.append(q[actions[i]])
            # Next state
            h2 = self.gnn(next_node_feats_batch[i], next_edge_index_batch[i])
            node_embed2 = h2[next_node_idx_batch[i]]
            q2 = self.target_qnet(node_embed2, next_dest_feat_batch[i])
            next_q_values.append(q2.max())
        q_values = torch.stack(q_values)
        next_q_values = torch.stack(next_q_values)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        targets = rewards + GAMMA * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Soft update target network
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return loss.item()

# --- Training Loop ---
def train():
    env = SimpleMANETEnv(NUM_NODES)
    node_feat_dim = 2 + 1 + NUM_NODES  # [x, y, queue, link qualities]
    dest_dim = 2 + 1  # [x, y, dst_id]
    action_dim = NUM_NODES
    agent = MADQNAgent(node_feat_dim, dest_dim, action_dim)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    epsilon = 1.0
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        for t in range(MAX_TTL):
            node_feats, edge_index, dest_feat, node_idx = state
            valid_actions = env.get_valid_actions(node_idx)
            if len(valid_actions) == 0:
                break
            epsilon = EPSILON_MIN + (1.0 - EPSILON_MIN) * np.exp(-episode / EPSILON_DECAY)
            action = agent.select_action(node_feats, edge_index, dest_feat, node_idx, valid_actions, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update(replay_buffer)
            state = next_state
            total_reward += reward
            if done:
                break
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Loss: {loss:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    train()