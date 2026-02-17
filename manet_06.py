import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv  # Added for temporal GNN
import numpy as np
import random
import matplotlib.pyplot as plt

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
    def __init__(self, max_neighbors):
        self.max_neighbors = max_neighbors
        self.q_values = {}  # Dynamic dict: neighbor_id -> q_value
        self.alpha = 0.3  # Higher learning rate for faster adaptation

    def forward(self, context, neighbor_ids):
        # context: [queue_length, success_ratio, energy] per neighbor
        # Simple linear combination for demonstration
        throughput_score = 1.0 / (1.0 + context[:, 0])  # inverse queue length
        throughput_score += context[:, 1]  # success ratio
        throughput_score += context[:, 2] * 0.1  # energy factor
        # Add learned q-values for known neighbors
        for i, nid in enumerate(neighbor_ids):
            if nid in self.q_values:
                throughput_score[i] += self.q_values[nid] * 0.5
        return throughput_score

    def update(self, neighbor_id, reward):
        # Online update for contextual bandit using neighbor_id (not index)
        if neighbor_id is None:
            return
        if neighbor_id not in self.q_values:
            self.q_values[neighbor_id] = 0.0
        self.q_values[neighbor_id] += self.alpha * (reward - self.q_values[neighbor_id])

# === Agent D: Exploration Agent (UCB) ===
class ExplorationAgent:
    def __init__(self, max_neighbors):
        self.max_neighbors = max_neighbors
        self.counts = {}  # Dynamic dict: neighbor_id -> count
        self.values = {}  # Dynamic dict: neighbor_id -> value
        self.total_count = 0
        self.ucb_c = 1.5  # Tuned exploration constant

    def forward(self, neighbor_ids):
        # UCB calculation for dynamic neighbors
        n_neighbors = len(neighbor_ids)
        exploration_bonus = np.zeros(n_neighbors)
        for i, nid in enumerate(neighbor_ids):
            count = self.counts.get(nid, 0)
            value = self.values.get(nid, 0.0)
            if count == 0:
                exploration_bonus[i] = 1e6  # force exploration
            else:
                exploration_bonus[i] = value + self.ucb_c * np.sqrt(2 * np.log(self.total_count + 1) / count)
        return exploration_bonus

    def update(self, neighbor_id, reward):
        # Update using neighbor_id (not index)
        if neighbor_id is None:
            return
        if neighbor_id not in self.counts:
            self.counts[neighbor_id] = 0
            self.values[neighbor_id] = 0.0
        self.counts[neighbor_id] += 1
        self.total_count += 1
        n = self.counts[neighbor_id]
        value = self.values[neighbor_id]
        self.values[neighbor_id] = ((n - 1) * value + reward) / n

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
            return packet.dst, ('direct', packet.dst)

        # Agent communication: share outputs, not raw data
        pdr_scores = self.agent_a(neighbor_graph).detach().cpu().numpy()
        delay_risks = self.agent_b(temporal_data).detach().cpu().numpy()
        throughput_scores = self.agent_c.forward(context_data, neighbor_ids)
        exploration_bonuses = self.agent_d.forward(neighbor_ids)

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
            return None, None
        next_hop_idx = np.argmax(final_score)
        next_hop = neighbor_ids[next_hop_idx]
        return next_hop, ('selected', next_hop)

    def update_agents(self, rewards, next_hop_id):
        self.agent_a.update(rewards['a'])
        self.agent_b.update(rewards['b'])
        self.agent_c.update(next_hop_id, rewards['c'])
        self.agent_d.update(next_hop_id, rewards['d'])

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
        'overhead': [],
        'delivered_count': [],
        'avg_delay': [],
        'throughput_mbps': []
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
        newly_delivered = []  # Track packets delivered this step
        for packet in packets:
            if packet.delivered:
                continue
            current_node = nodes[packet.hops[-1]]
            neighbor_ids = current_node.agent_e.neighbors
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
                packet.hops.append(next_hop)
                # Check if delivered
                if next_hop == packet.dst:
                    packet.delivered = True
                    packet.delivered_time = t
                    delivered_packets.append(packet)
                    newly_delivered.append(packet)
                    delivery_reward = 1.0
                else:
                    delivery_reward = 0.1  # Small reward for progress
                
                # Calculate delay penalty (lower is better)
                hops_so_far = len(packet.hops) - 1
                delay_reward = max(0, 1.0 - hops_so_far * 0.1)
                
                # Rewards for agent updates
                rewards = {
                    'a': delivery_reward,
                    'b': delay_reward,
                    'c': delivery_reward,
                    'd': delivery_reward
                }
                current_node.update_agents(rewards, next_hop)
        # Federated learning step
        if t > 0 and t % fed_interval == 0:
            federated_learning(nodes)
        # Metrics
        delivered = [p for p in packets if p.delivered]
        pdr = len(delivered) / len(packets) if packets else 0
        metrics['pdr'].append(pdr)
        
        # Track delivered count over time
        metrics['delivered_count'].append(len(delivered))
        
        # Track average delay over time
        if delivered:
            current_avg_delay = np.mean([p.delivered_time - p.created_time for p in delivered])
        else:
            current_avg_delay = 0.0
        metrics['avg_delay'].append(current_avg_delay)
        
        # Track throughput over time (Mbps)
        total_bits = len(delivered) * PACKET_SIZE_BITS
        current_throughput = total_bits / (t + 1) / 1e6 if (t + 1) > 0 else 0.0
        metrics['throughput_mbps'].append(current_throughput)
        
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
    
    # === Plot Metrics ===
    plot_metrics(metrics, num_steps)

def plot_metrics(metrics, num_steps):
    """Plot delivered packets, avg PDR, avg delay, and throughput in a 2x2 grid."""
    time_steps = list(range(num_steps))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MANET Multi-Agent Routing Simulation Metrics', fontsize=14, fontweight='bold')
    
    # Plot 1: Delivered Packets Over Time
    ax1 = axes[0, 0]
    ax1.plot(time_steps, metrics['delivered_count'], color='green', linewidth=1.5)
    ax1.set_xlabel('Time Step (s)')
    ax1.set_ylabel('Delivered Packets')
    ax1.set_title('Delivered Packets Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.fill_between(time_steps, metrics['delivered_count'], alpha=0.3, color='green')
    
    # Plot 2: Average PDR Over Time
    ax2 = axes[0, 1]
    ax2.plot(time_steps, metrics['pdr'], color='blue', linewidth=1.5)
    ax2.set_xlabel('Time Step (s)')
    ax2.set_ylabel('Packet Delivery Ratio (PDR)')
    ax2.set_title('Average PDR Over Time')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.fill_between(time_steps, metrics['pdr'], alpha=0.3, color='blue')
    
    # Plot 3: Average Delay Over Time
    ax3 = axes[1, 0]
    ax3.plot(time_steps, metrics['avg_delay'], color='red', linewidth=1.5)
    ax3.set_xlabel('Time Step (s)')
    ax3.set_ylabel('Delay (seconds)')
    ax3.set_title('Average Delay Over Time')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.fill_between(time_steps, metrics['avg_delay'], alpha=0.3, color='red')
    
    # Plot 4: Throughput Over Time
    ax4 = axes[1, 1]
    ax4.plot(time_steps, metrics['throughput_mbps'], color='purple', linewidth=1.5)
    ax4.set_xlabel('Time Step (s)')
    ax4.set_ylabel('Throughput (Mbps)')
    ax4.set_title('Throughput Over Time')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.fill_between(time_steps, metrics['throughput_mbps'], alpha=0.3, color='purple')
    
    plt.tight_layout()
    plt.savefig('manet_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Metrics plot saved as 'manet_metrics.png'")

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
    # Aggregate q_values (dict-based) across all nodes
    all_keys = set()
    for node in nodes:
        all_keys.update(node.agent_c.q_values.keys())
    
    if all_keys:
        avg_q_values = {}
        for key in all_keys:
            values = [n.agent_c.q_values.get(key, 0.0) for n in nodes]
            avg_q_values[key] = np.mean(values)
        # Broadcast averaged q_values to all nodes
        for node in nodes:
            node.agent_c.q_values = avg_q_values.copy()
    
    # --- Federated learning for ExplorationAgent (Agent D) ---
    all_neighbor_ids = set()
    for node in nodes:
        all_neighbor_ids.update(node.agent_d.values.keys())
    
    if all_neighbor_ids:
        avg_values = {}
        avg_counts = {}
        for nid in all_neighbor_ids:
            values = [n.agent_d.values.get(nid, 0.0) for n in nodes]
            counts = [n.agent_d.counts.get(nid, 0) for n in nodes]
            avg_values[nid] = np.mean(values)
            avg_counts[nid] = int(np.mean(counts))
        # Broadcast to all nodes
        for node in nodes:
            node.agent_d.values = avg_values.copy()
            node.agent_d.counts = avg_counts.copy()

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