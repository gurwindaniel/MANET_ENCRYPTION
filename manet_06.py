import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv  # Added for temporal GNN
import numpy as np
import random

# === Agent A: Reliability Agent (GNN) ===
class ReliabilityAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = 0.003  # Lower learning rate for stability in federated setting

    def forward(self, data):
        # data: PyTorch Geometric Data object
        x, edge_index = data.x, data.edge_index
        h = self.gnn(x, edge_index)
        out = self.fc(h)
        pdr_score = self.sigmoid(out).squeeze(-1)
        return pdr_score

    def update(self, reward):
        # Online learning step (to be implemented)
        pass

# === Agent B: Delay Agent (Temporal GNN) ===
class DelayAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=5):  # More layers for better modeling
        super().__init__()
        self.temporal_gnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.lr = 0.003  # Lower learning rate for stability in federated setting

    def forward(self, data):
        # data: PyTorch Geometric Data object with temporal features
        x, edge_index = data.x, data.edge_index
        # GatedGraphConv expects x: [N, in_channels], edge_index: [2, E]
        h = self.temporal_gnn(x, edge_index)
        delay_risk = self.fc(h).squeeze(-1)
        return delay_risk

    def update(self, reward):
        # Online learning step (to be implemented)
        pass

# === Agent C: Throughput Agent (Contextual Bandit) ===
class ThroughputAgent:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.q_values = np.zeros(n_neighbors)
        self.alpha = 0.3  # Higher learning rate for faster adaptation

    def forward(self, context):
        # context: [queue_length, success_ratio, energy] per neighbor
        # Simple linear combination for demonstration
        throughput_score = 1.0 / (1.0 + context[:, 0])  # inverse queue length
        throughput_score += context[:, 1]  # success ratio
        throughput_score += context[:, 2] * 0.1  # energy factor
        return throughput_score

    def update(self, neighbor_idx, reward):
        # Online update for contextual bandit
        self.q_values[neighbor_idx] += self.alpha * (reward - self.q_values[neighbor_idx])

# === Agent D: Exploration Agent (UCB) ===
class ExplorationAgent:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.counts = np.zeros(n_neighbors)
        self.values = np.zeros(n_neighbors)
        self.total_count = 0
        self.ucb_c = 1.5  # Tuned exploration constant

    def forward(self):
        # UCB calculation
        exploration_bonus = np.zeros(self.n_neighbors)
        for i in range(self.n_neighbors):
            if self.counts[i] == 0:
                exploration_bonus[i] = 1e6  # force exploration
            else:
                exploration_bonus[i] = self.values[i] + self.ucb_c * np.sqrt(2 * np.log(self.total_count + 1) / self.counts[i])
        return exploration_bonus

    def update(self, neighbor_idx, reward):
        self.counts[neighbor_idx] += 1
        self.total_count += 1
        n = self.counts[neighbor_idx]
        value = self.values[neighbor_idx]
        self.values[neighbor_idx] = ((n - 1) * value + reward) / n

# === Agent E: Neighbor Tracking Agent ===
class NeighborTrackingAgent:
    def __init__(self, node, comm_range):
        self.node = node
        self.comm_range = comm_range
        self.neighbors = []

    def update_neighbors(self, all_nodes):
        self.neighbors = []
        for other in all_nodes:
            if other.node_id != self.node.node_id:
                dist = np.linalg.norm(self.node.position - other.position)
                if dist <= self.comm_range:
                    self.neighbors.append(other.node_id)
        return self.neighbors

# === Data Packet ===
class DataPacket:
    def __init__(self, src, dst, created_time):
        self.src = src
        self.dst = dst
        self.created_time = created_time
        self.hops = [src]
        self.delivered = False
        self.delivered_time = None

# === Node Class (updated) ===
class Node:
    def __init__(self, node_id, n_neighbors, agent_dims, fusion_weights, comm_range, area_size):
        self.node_id = node_id
        self.n_neighbors = n_neighbors
        self.agent_a = ReliabilityAgent(agent_dims['a_in'], agent_dims['a_hidden'])
        self.agent_b = DelayAgent(agent_dims['b_in'], agent_dims['b_hidden'])
        self.agent_c = ThroughputAgent(n_neighbors)
        self.agent_d = ExplorationAgent(n_neighbors)
        self.agent_e = NeighborTrackingAgent(self, comm_range)
        self.fusion_weights = fusion_weights
        self.position = np.random.rand(2) * area_size  # (x, y)
        self.area_size = area_size
        self.comm_range = comm_range
        self.mobility_target = self._random_point()
        self.speed = random.uniform(10, 18)  # Higher speed for faster delivery

    def _random_point(self):
        return np.random.rand(2) * self.area_size

    def move(self, dt=1.0):
        direction = self.mobility_target - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-2:
            self.mobility_target = self._random_point()
            direction = self.mobility_target - self.position
            dist = np.linalg.norm(direction)
        if dist > 0:
            step = min(self.speed * dt, dist)
            self.position += direction / dist * step

    def update_neighbors(self, all_nodes):
        return self.agent_e.update_neighbors(all_nodes)

    def select_next_hop(self, neighbor_graph, temporal_data, context_data, neighbor_ids, packet=None):
        # Prefer direct delivery if possible
        if packet and packet.dst in neighbor_ids:
            return packet.dst, None

        # Agent communication: share outputs, not raw data
        pdr_scores = self.agent_a(neighbor_graph).detach().cpu().numpy()
        delay_risks = self.agent_b(temporal_data).detach().cpu().numpy()
        throughput_scores = self.agent_c.forward(context_data)
        exploration_bonuses = self.agent_d.forward()

        original_neighbor_ids = neighbor_ids.copy()  # Save original for index mapping

        n_neighbors = len(neighbor_ids)  # Always set n_neighbors to current neighbor_ids length

        # Truncate or pad agent outputs to match n_neighbors BEFORE filtering
        def align(arr):
            arr = np.asarray(arr)
            if arr.shape[0] > n_neighbors:
                return arr[:n_neighbors]
            elif arr.shape[0] < n_neighbors:
                return np.pad(arr, (0, n_neighbors - arr.shape[0]), constant_values=-np.inf)
            return arr

        pdr_scores = align(pdr_scores)
        delay_risks = align(delay_risks)
        throughput_scores = align(throughput_scores)
        exploration_bonuses = align(exploration_bonuses)

        # Avoid routing loops: don't revisit nodes already in hops
        if packet:
            valid_indices = [i for i, nid in enumerate(neighbor_ids) if nid not in packet.hops]
            if not valid_indices:
                return None, None
            # Filter scores to only valid neighbors
            pdr_scores = pdr_scores[valid_indices]
            delay_risks = delay_risks[valid_indices]
            throughput_scores = throughput_scores[valid_indices]
            exploration_bonuses = exploration_bonuses[valid_indices]
            neighbor_ids = [neighbor_ids[i] for i in valid_indices]
            n_neighbors = len(neighbor_ids)  # Update n_neighbors after filtering

        alpha = self.fusion_weights['alpha']
        beta = self.fusion_weights['beta']
        gamma = self.fusion_weights['gamma']
        delta = self.fusion_weights['delta']
        final_score = (
            alpha * pdr_scores
            - beta * delay_risks
            + gamma * throughput_scores
            + delta * exploration_bonuses
        )

        if n_neighbors == 0 or np.all(np.isneginf(final_score)):
            return None, final_score
        next_hop_idx = np.argmax(final_score)
        next_hop = neighbor_ids[next_hop_idx]
        orig_idx = original_neighbor_ids.index(next_hop)
        return next_hop, (final_score, orig_idx)

    def update_agents(self, rewards):
        self.agent_a.update(rewards['a'])
        self.agent_b.update(rewards['b'])
        self.agent_c.update(rewards['c']['idx'], rewards['c']['reward'])
        self.agent_d.update(rewards['d']['idx'], rewards['d']['reward'])

# === Steady-State Random Waypoint Mobility ===
def update_node_positions(nodes, dt=1.0):
    for node in nodes:
        node.move(dt=dt)

# === Simulation Loop ===
def simulate(nodes, num_steps, area_size):
    metrics = {
        'pdr': [],
        'throughput': [],
        'delay': [],
        'energy': [],
        'overhead': []
    }
    comm_range = nodes[0].comm_range
    packets = []
    delivered_packets = []
    fed_interval = 40  # Federated learning every 40 steps
    PACKET_SIZE_BYTES = 4096  # Increased packet size (was 1024)
    PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8
    for t in range(num_steps):
        # Move nodes
        update_node_positions(nodes)
        # Update neighbors for all nodes
        for node in nodes:
            node.update_neighbors(nodes)
        # Generate more packets per step for higher throughput
        if t % 5 == 0:
            for _ in range(10):  # Increased: Generate 10 packets per interval (was 2)
                src = random.choice(nodes)
                dst = random.choice([n for n in nodes if n.node_id != src.node_id])
                packet = DataPacket(src.node_id, dst.node_id, t)
                packets.append(packet)
        # Forward packets
        for packet in packets:
            if packet.delivered:
                continue
            current_node = nodes[packet.hops[-1]]
            neighbor_ids = current_node.agent_e.neighbors
            # Prefer direct delivery
            if packet.dst in neighbor_ids:
                packet.hops.append(packet.dst)
                packet.delivered = True
                packet.delivered_time = t
                delivered_packets.append(packet)
                continue
            if not neighbor_ids:
                continue
            n_neighbors = len(neighbor_ids)
            neighbor_graph = Data(
                x=torch.randn(n_neighbors, 8),
                edge_index=torch.randint(0, n_neighbors, (2, n_neighbors*2))
            )
            temporal_data = Data(
                x=torch.randn(n_neighbors, 8),
                edge_index=torch.randint(0, n_neighbors, (2, n_neighbors*2))
            )
            context_data = np.random.rand(n_neighbors, 3)
            # Pass packet to avoid loops and prefer direct delivery
            next_hop, result = current_node.select_next_hop(
                neighbor_graph, temporal_data, context_data, neighbor_ids, packet=packet
            )
            if next_hop is not None:
                # result is (final_score, orig_idx)
                _, orig_idx = result if result is not None else (None, None)
                packet.hops.append(next_hop)
                # Dummy rewards for agent updates
                rewards = {
                    'a': 1 if packet.delivered else 0,
                    'b': 1.0,
                    'c': {'idx': orig_idx, 'reward': 1.0},
                    'd': {'idx': orig_idx, 'reward': 1.0}
                }
                current_node.update_agents(rewards)
        # Federated learning step
        if t > 0 and t % fed_interval == 0:
            federated_learning(nodes)
        # Metrics
        delivered = [p for p in packets if p.delivered]
        pdr = len(delivered) / len(packets) if packets else 0
        metrics['pdr'].append(pdr)
        # ...other metrics...
    # Print summary
    print(f"Delivered: {len(delivered_packets)}/{len(packets)}")
    print(f"Avg PDR: {np.mean(metrics['pdr']) if metrics['pdr'] else 0:.3f}")

    # --- Modified: Print delay in seconds and throughput in Mbps ---
    if delivered_packets:
        # Each step is 1 second, so delay in seconds
        avg_delay = np.mean([p.delivered_time - p.created_time for p in delivered_packets])  # seconds
    else:
        avg_delay = 0.0

    # Throughput in Mbps: (total bits delivered) / (total time in seconds) / 1e6
    total_bits = len(delivered_packets) * PACKET_SIZE_BITS
    throughput_mbps = total_bits / num_steps / 1e6 if num_steps > 0 else 0.0

    print(f"Avg Delay (s): {avg_delay:.3f}")
    print(f"Throughput (Mbps): {throughput_mbps:.6f}")

# --- Federated Learning Utilities ---
def federated_average(models):
    """Average the parameters of a list of PyTorch models (assumes same architecture)."""
    avg_state = {}
    n = len(models)
    for k in models[0].state_dict().keys():
        avg_state[k] = sum([m.state_dict()[k].float() for m in models]) / n
    for m in models:
        m.load_state_dict(avg_state)
    return

def federated_average_numpy(arrays):
    """Average a list of numpy arrays (for non-PyTorch agents)."""
    avg = np.mean(np.stack(arrays), axis=0)
    for arr in arrays:
        arr[:] = avg  # in-place update

def federated_learning(nodes):
    # Aggregate and broadcast for ReliabilityAgent and DelayAgent
    reliability_agents = [n.agent_a for n in nodes]
    delay_agents = [n.agent_b for n in nodes]
    federated_average(reliability_agents)
    federated_average(delay_agents)

    # --- Federated learning for ThroughputAgent (Agent C) ---
    throughput_agents = [n.agent_c for n in nodes]
    q_values_list = [agent.q_values for agent in throughput_agents]
    federated_average_numpy(q_values_list)
    # Ensure all agents' q_values are updated
    for agent, avg_q in zip(throughput_agents, q_values_list):
        agent.q_values = avg_q

# === Example Usage ===
if __name__ == "__main__":
    n_nodes = 100
    n_neighbors = 10  # More neighbors for more routing options
    area_size = 1000   # Smaller area for higher node density
    comm_range = 250
    agent_dims = {
        'a_in': 8, 'a_hidden': 32,   # Larger hidden for better capacity
        'b_in': 8, 'b_hidden': 32
    }
    # Fusion weights: penalize delay more, reward throughput more
    fusion_weights = {'alpha': 1.5, 'beta': 2.5, 'gamma': 4.0, 'delta': 1.0}
    nodes = [
        Node(i, n_neighbors, agent_dims, fusion_weights, comm_range, area_size)
        for i in range(n_nodes)
    ]
    simulate(nodes, num_steps=400, area_size=area_size)