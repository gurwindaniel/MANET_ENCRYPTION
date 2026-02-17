"""
MANET Opportunistic Routing - Multi-Configuration Experiments
Test configurations: Nodes = [100, 200, 300, 400, 500], Speeds = [20, 25, 30, 35, 40]
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AGENTS (Same as manet_06_opportunistic.py)
# ============================================================================

class ReliabilityAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.gnn(x, edge_index)
        out = self.fc(h)
        return self.sigmoid(out).squeeze(-1)

    def update(self, reward):
        pass


class CandidateForwarderTGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.temporal_gnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fc_priority = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.contact_history = defaultdict(list)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.temporal_gnn(x, edge_index)
        attention_weights = self.sigmoid(self.attention(h))
        priority_scores = self.sigmoid(self.fc_priority(h)).squeeze(-1)
        return priority_scores * attention_weights.squeeze(-1)

    def update_contact_history(self, neighbor_id, current_time, in_contact):
        history = self.contact_history[neighbor_id]
        if in_contact:
            if not history or history[-1][1] is not None:
                history.append([current_time, None])
        else:
            if history and history[-1][1] is None:
                history[-1][1] = current_time

    def update(self, reward):
        pass


class ThroughputAgent:
    def __init__(self, max_neighbors):
        self.q_values = {}
        self.alpha = 0.3

    def forward(self, context, neighbor_ids):
        throughput_score = 1.0 / (1.0 + context[:, 0])
        throughput_score += context[:, 1]
        throughput_score += context[:, 2] * 0.1
        for i, nid in enumerate(neighbor_ids):
            if nid in self.q_values:
                throughput_score[i] += self.q_values[nid] * 0.5
        return throughput_score

    def update(self, neighbor_id, reward):
        if neighbor_id is None:
            return
        if neighbor_id not in self.q_values:
            self.q_values[neighbor_id] = 0.0
        self.q_values[neighbor_id] += self.alpha * (reward - self.q_values[neighbor_id])


class ExplorationAgent:
    def __init__(self, max_neighbors):
        self.counts = {}
        self.values = {}
        self.total_count = 0
        self.ucb_c = 1.5

    def forward(self, neighbor_ids):
        n_neighbors = len(neighbor_ids)
        exploration_bonus = np.zeros(n_neighbors)
        for i, nid in enumerate(neighbor_ids):
            count = self.counts.get(nid, 0)
            value = self.values.get(nid, 0.0)
            if count == 0:
                exploration_bonus[i] = 1e6
            else:
                exploration_bonus[i] = value + self.ucb_c * np.sqrt(2 * np.log(self.total_count + 1) / count)
        return exploration_bonus

    def update(self, neighbor_id, reward):
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


class NeighborTrackingAgent:
    def __init__(self, node, comm_range):
        self.node = node
        self.comm_range = comm_range
        self.neighbors = []
        self.neighbor_metrics = {}

    def update_neighbors(self, all_nodes, current_time, dt=1.0):
        self.neighbors = []
        for other in all_nodes:
            if other.node_id == self.node.node_id:
                continue
            dist = np.linalg.norm(self.node.position - other.position)
            if dist <= self.comm_range:
                self.neighbors.append(other.node_id)
                # Simplified metrics calculation
                rel_speed = np.linalg.norm(other.velocity - self.node.velocity) if hasattr(other, 'velocity') else 0
                self.neighbor_metrics[other.node_id] = {
                    'distance': dist,
                    'relative_speed': rel_speed,
                    'approach_rate': 0.0,
                    'link_stability': 1.0 / (1.0 + rel_speed * 0.1),
                    'contact_duration': max(10, (self.comm_range - dist) / (rel_speed + 0.1)),
                    'direction_to_other': np.zeros(2)
                }
        return self.neighbors

    def _compute_neighbor_metrics(self, other_node, dist, dt):
        my_velocity = self.node.velocity if hasattr(self.node, 'velocity') else np.zeros(2)
        other_velocity = other_node.velocity if hasattr(other_node, 'velocity') else np.zeros(2)
        relative_velocity = other_velocity - my_velocity
        relative_speed = np.linalg.norm(relative_velocity)
        
        direction_to_other = (other_node.position - self.node.position)
        norm = np.linalg.norm(direction_to_other)
        if norm > 1e-6:
            direction_to_other = direction_to_other / norm
        else:
            direction_to_other = np.zeros(2)
        
        approach_rate = -np.dot(relative_velocity, direction_to_other)
        
        if relative_speed > 0.1:
            remaining_dist = self.comm_range - dist
            contact_duration = max(0, remaining_dist / relative_speed) if approach_rate <= 0 else 100.0
        else:
            contact_duration = 100.0
        
        return {
            'distance': dist,
            'relative_speed': relative_speed,
            'approach_rate': approach_rate,
            'link_stability': 1.0 / (1.0 + abs(approach_rate)),
            'contact_duration': contact_duration,
            'direction_to_other': direction_to_other
        }

    def get_metrics_for_neighbor(self, neighbor_id):
        return self.neighbor_metrics.get(neighbor_id, None)


class DataPacket:
    def __init__(self, src, dst, created_time, packet_id=None):
        self.packet_id = packet_id or f"{src}_{dst}_{created_time}"
        self.src = src
        self.dst = dst
        self.created_time = created_time
        self.hops = [src]
        self.delivered = False
        self.delivered_time = None
        self.current_holder = src
        self.forwarded_to = set()
        self.ttl = 50


class Node:
    def __init__(self, node_id, n_neighbors, agent_dims, fusion_weights, comm_range, area_size, speed):
        self.node_id = node_id
        self.n_neighbors = n_neighbors
        self.agent_a = ReliabilityAgent(agent_dims['a_in'], agent_dims['a_hidden'])
        self.agent_b = CandidateForwarderTGNN(agent_dims['b_in'], agent_dims['b_hidden'])
        self.agent_c = ThroughputAgent(n_neighbors)
        self.agent_d = ExplorationAgent(n_neighbors)
        self.agent_e = NeighborTrackingAgent(self, comm_range)
        self.fusion_weights = fusion_weights
        self.position = np.random.rand(2) * area_size
        self.velocity = np.zeros(2)
        self.prev_position = self.position.copy()
        self.area_size = area_size
        self.comm_range = comm_range
        self.mobility_target = self._random_point()
        self.speed = speed  # Fixed speed for experiments
        self.packet_buffer = []
        self.max_buffer_size = 100
        self.delivery_history = defaultdict(list)

    def _random_point(self):
        return np.random.rand(2) * self.area_size

    def move(self, dt=1.0):
        self.prev_position = self.position.copy()
        direction = self.mobility_target - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-2:
            self.mobility_target = self._random_point()
            direction = self.mobility_target - self.position
            dist = np.linalg.norm(direction)
        if dist > 0:
            step = min(self.speed * dt, dist)
            self.position += direction / dist * step
        self.velocity = (self.position - self.prev_position) / dt

    def update_neighbors(self, all_nodes, current_time):
        return self.agent_e.update_neighbors(all_nodes, current_time)

    def buffer_packet(self, packet):
        if len(self.packet_buffer) < self.max_buffer_size:
            packet.current_holder = self.node_id
            self.packet_buffer.append(packet)
            return True
        return False

    def get_candidate_forwarders(self, all_nodes, packet, current_time, top_k=5):
        neighbor_ids = self.agent_e.neighbors
        if not neighbor_ids:
            return []
        
        if packet.dst in neighbor_ids:
            return [(packet.dst, float('inf'))]
        
        valid_neighbors = [nid for nid in neighbor_ids 
                         if nid not in packet.hops and nid not in packet.forwarded_to]
        
        if not valid_neighbors:
            return []
        
        n_neighbors = len(valid_neighbors)
        temporal_features = []
        dst_node = all_nodes[packet.dst]
        
        for nid in valid_neighbors:
            metrics = self.agent_e.get_metrics_for_neighbor(nid)
            neighbor_node = all_nodes[nid]
            
            if metrics:
                rel_speed = metrics['relative_speed']
                link_stability = metrics['link_stability']
                contact_duration = metrics['contact_duration']
                
                dist_to_dst = np.linalg.norm(neighbor_node.position - dst_node.position)
                my_dist_to_dst = np.linalg.norm(self.position - dst_node.position)
                dist_ratio = dist_to_dst / (my_dist_to_dst + 1e-6)
                
                dir_to_dst = dst_node.position - neighbor_node.position
                if np.linalg.norm(dir_to_dst) > 0:
                    dir_to_dst = dir_to_dst / np.linalg.norm(dir_to_dst)
                neighbor_vel = neighbor_node.velocity
                if np.linalg.norm(neighbor_vel) > 0:
                    neighbor_vel_norm = neighbor_vel / np.linalg.norm(neighbor_vel)
                    direction_alignment = np.dot(neighbor_vel_norm, dir_to_dst)
                else:
                    direction_alignment = 0.0
                
                hist = self.delivery_history.get(nid, [])
                success_rate = np.mean(hist) if hist else 0.5
                energy = random.uniform(0.5, 1.0)
                queue_len = len(neighbor_node.packet_buffer) / neighbor_node.max_buffer_size
                
                temporal_features.append([
                    rel_speed / 50.0,
                    link_stability,
                    min(contact_duration, 100) / 100.0,
                    1.0 - min(dist_ratio, 2.0) / 2.0,
                    (direction_alignment + 1) / 2.0,
                    energy,
                    1.0 - queue_len,
                    success_rate
                ])
            else:
                temporal_features.append([0.5] * 8)
        
        temporal_features = np.array(temporal_features, dtype=np.float32)
        edge_index = self._create_neighbor_graph_edges(n_neighbors)
        
        neighbor_graph = Data(
            x=torch.tensor(temporal_features, dtype=torch.float32),
            edge_index=edge_index
        )
        
        with torch.no_grad():
            pdr_scores = self.agent_a(neighbor_graph).cpu().numpy()
            tgnn_scores = self.agent_b(neighbor_graph).cpu().numpy()
        
        context_data = temporal_features[:, [6, 7, 5]]
        throughput_scores = self.agent_c.forward(context_data, valid_neighbors)
        exploration_bonuses = self.agent_d.forward(valid_neighbors)
        
        def align(arr):
            arr = np.asarray(arr)
            if arr.shape[0] > n_neighbors:
                return arr[:n_neighbors]
            elif arr.shape[0] < n_neighbors:
                return np.pad(arr, (0, n_neighbors - arr.shape[0]), constant_values=0)
            return arr
        
        pdr_scores = align(pdr_scores)
        tgnn_scores = align(tgnn_scores)
        throughput_scores = align(throughput_scores)
        exploration_bonuses = align(exploration_bonuses)
        
        alpha = self.fusion_weights['alpha']
        beta = self.fusion_weights['beta']
        gamma = self.fusion_weights['gamma']
        delta = self.fusion_weights['delta']
        
        distance_progress = temporal_features[:, 3]
        link_stability = temporal_features[:, 1]
        contact_duration = temporal_features[:, 2]
        low_queue = temporal_features[:, 6]
        
        final_scores = (
            alpha * pdr_scores
            + beta * tgnn_scores
            + gamma * throughput_scores
            + delta * exploration_bonuses
            + 4.0 * distance_progress
            + 1.5 * link_stability
            + 1.0 * contact_duration
            + 0.5 * low_queue
        )
        
        candidates = [(valid_neighbors[i], final_scores[i]) for i in range(n_neighbors)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _create_neighbor_graph_edges(self, n_neighbors):
        if n_neighbors <= 1:
            return torch.zeros((2, 0), dtype=torch.long)
        edges = []
        for i in range(n_neighbors):
            for j in range(n_neighbors):
                if i != j:
                    edges.append([i, j])
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        return torch.zeros((2, 0), dtype=torch.long)

    def update_agents(self, rewards, next_hop_id):
        self.agent_a.update(rewards['a'])
        self.agent_b.update(rewards['b'])
        self.agent_c.update(next_hop_id, rewards['c'])
        self.agent_d.update(next_hop_id, rewards['d'])

    def record_delivery_result(self, neighbor_id, success):
        self.delivery_history[neighbor_id].append(1.0 if success else 0.0)
        if len(self.delivery_history[neighbor_id]) > 100:
            self.delivery_history[neighbor_id] = self.delivery_history[neighbor_id][-100:]


def update_node_positions(nodes, dt=1.0):
    for node in nodes:
        node.move(dt=dt)


def run_single_experiment(n_nodes, node_speed, num_steps=50):
    """Run a single experiment with given node count and speed."""
    # Scale area with node count for consistent density
    area_size = int(np.sqrt(n_nodes) * 70)  # Smaller area for speed
    comm_range = int(area_size * 0.4)  # Higher range for connectivity
    
    agent_dims = {'a_in': 8, 'a_hidden': 16, 'b_in': 8, 'b_hidden': 16}  # Minimal for speed
    fusion_weights = {'alpha': 1.5, 'beta': 2.5, 'gamma': 2.0, 'delta': 0.3}
    
    nodes = [
        Node(i, 15, agent_dims, fusion_weights, comm_range, area_size, node_speed)
        for i in range(n_nodes)
    ]
    
    packets = []
    delivered_packets = []
    delivered_packet_ids = set()
    PACKET_SIZE_BYTES = 8192
    PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8
    packet_counter = 0
    FORWARDING_ROUNDS = 1  # Single round for speed
    MAX_PACKETS_PER_CONTACT = 2  # Minimal for speed
    
    for t in range(num_steps):
        update_node_positions(nodes)
        for node in nodes:
            node.update_neighbors(nodes, t)
        
        # Generate packets proportional to node count
        if t % 3 == 0:
            packets_to_gen = max(5, n_nodes // 20)
            for _ in range(packets_to_gen):
                src = random.choice(nodes)
                dst = random.choice([n for n in nodes if n.node_id != src.node_id])
                packet = DataPacket(src.node_id, dst.node_id, t, packet_id=packet_counter)
                packet_counter += 1
                packets.append(packet)
                src.buffer_packet(packet)
        
        # Opportunistic forwarding
        for _ in range(FORWARDING_ROUNDS):
            for node in nodes:
                packets_to_remove = []
                forwarded_count = 0
                
                sorted_buffer = sorted(
                    node.packet_buffer,
                    key=lambda p: (t - p.created_time, -len(p.hops)),
                    reverse=True
                )
                
                for packet in sorted_buffer:
                    if forwarded_count >= MAX_PACKETS_PER_CONTACT:
                        break
                    
                    if packet.delivered or packet.packet_id in delivered_packet_ids:
                        packets_to_remove.append(packet)
                        continue
                    
                    if packet.ttl <= 0:
                        packets_to_remove.append(packet)
                        continue
                    
                    candidates = node.get_candidate_forwarders(nodes, packet, t, top_k=5)
                    if not candidates:
                        continue
                    
                    for candidate_id, score in candidates:
                        candidate_node = nodes[candidate_id]
                        
                        if candidate_id == packet.dst:
                            packet.hops.append(candidate_id)
                            packet.delivered = True
                            packet.delivered_time = t
                            delivered_packets.append(packet)
                            delivered_packet_ids.add(packet.packet_id)
                            packets_to_remove.append(packet)
                            
                            delay = t - packet.created_time
                            delay_bonus = max(0, 1.0 - delay * 0.1)
                            rewards = {'a': 1.0 + delay_bonus, 'b': 1.0 + delay_bonus, 
                                      'c': 1.0 + delay_bonus, 'd': 1.0 + delay_bonus}
                            node.update_agents(rewards, candidate_id)
                            node.record_delivery_result(candidate_id, True)
                            forwarded_count += 1
                            break
                        
                        if candidate_node.buffer_packet(packet):
                            packet.hops.append(candidate_id)
                            packet.forwarded_to.add(candidate_id)
                            packet.ttl -= 1
                            
                            dst_node = nodes[packet.dst]
                            old_dist = np.linalg.norm(node.position - dst_node.position)
                            new_dist = np.linalg.norm(candidate_node.position - dst_node.position)
                            progress_reward = max(0, (old_dist - new_dist) / old_dist) if old_dist > 0 else 0
                            
                            rewards = {'a': 0.4 + progress_reward * 0.6, 'b': 0.4 + progress_reward * 0.6,
                                      'c': 0.4 + progress_reward * 0.6, 'd': 0.4 + progress_reward * 0.6}
                            node.update_agents(rewards, candidate_id)
                            packets_to_remove.append(packet)
                            forwarded_count += 1
                            break
                
                for pkt in packets_to_remove:
                    if pkt in node.packet_buffer:
                        node.packet_buffer.remove(pkt)
    
    # Calculate metrics
    total_packets = len(packets)
    total_delivered = len(delivered_packets)
    pdr = total_delivered / total_packets if total_packets > 0 else 0
    
    if delivered_packets:
        avg_delay = np.mean([p.delivered_time - p.created_time for p in delivered_packets])
        avg_hops = np.mean([len(p.hops) - 1 for p in delivered_packets])
    else:
        avg_delay = num_steps
        avg_hops = 0
    
    throughput_mbps = (total_delivered * PACKET_SIZE_BITS) / num_steps / 1e6
    
    return {
        'pdr': pdr,
        'delay': avg_delay,
        'throughput': throughput_mbps,
        'hops': avg_hops,
        'delivered': total_delivered,
        'total': total_packets
    }


def run_all_experiments():
    """Run experiments for all node counts and speeds."""
    node_counts = [100, 200, 300, 400, 500]
    speeds = [20, 25, 30, 35, 40]
    num_steps = 50  # Reduced for faster execution
    
    # Results storage
    results = {
        'pdr': np.zeros((len(node_counts), len(speeds))),
        'delay': np.zeros((len(node_counts), len(speeds))),
        'throughput': np.zeros((len(node_counts), len(speeds))),
        'hops': np.zeros((len(node_counts), len(speeds)))
    }
    
    total_experiments = len(node_counts) * len(speeds)
    current = 0
    
    print("=" * 70)
    print("MANET OPPORTUNISTIC ROUTING - MULTI-CONFIGURATION EXPERIMENTS")
    print("=" * 70)
    print(f"Node counts: {node_counts}")
    print(f"Speeds: {speeds}")
    print(f"Steps per experiment: {num_steps}")
    print("-" * 70)
    
    for i, n_nodes in enumerate(node_counts):
        for j, speed in enumerate(speeds):
            current += 1
            print(f"\n[{current}/{total_experiments}] Running: Nodes={n_nodes}, Speed={speed} m/s")
            
            start_time = time.time()
            result = run_single_experiment(n_nodes, speed, num_steps)
            elapsed = time.time() - start_time
            
            results['pdr'][i, j] = result['pdr']
            results['delay'][i, j] = result['delay']
            results['throughput'][i, j] = result['throughput']
            results['hops'][i, j] = result['hops']
            
            print(f"   PDR: {result['pdr']:.3f} | Delay: {result['delay']:.3f}s | "
                  f"Throughput: {result['throughput']:.4f} Mbps | Hops: {result['hops']:.2f} | "
                  f"Time: {elapsed:.1f}s")
    
    return results, node_counts, speeds


def plot_results(results, node_counts, speeds):
    """Create comprehensive visualization of results."""
    
    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(18, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(speeds)))
    markers = ['o', 's', '^', 'D', 'v']
    
    # ===== Plot 1: PDR vs Number of Nodes =====
    ax1 = fig.add_subplot(2, 3, 1)
    for j, speed in enumerate(speeds):
        ax1.plot(node_counts, results['pdr'][:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'Speed={speed} m/s')
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Packet Delivery Ratio (PDR)', fontsize=12)
    ax1.set_title('PDR vs Number of Nodes', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    
    # ===== Plot 2: Delay vs Number of Nodes =====
    ax2 = fig.add_subplot(2, 3, 2)
    for j, speed in enumerate(speeds):
        ax2.plot(node_counts, results['delay'][:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'Speed={speed} m/s')
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Average Delay (seconds)', fontsize=12)
    ax2.set_title('Delay vs Number of Nodes', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # ===== Plot 3: Throughput vs Number of Nodes =====
    ax3 = fig.add_subplot(2, 3, 3)
    for j, speed in enumerate(speeds):
        ax3.plot(node_counts, results['throughput'][:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'Speed={speed} m/s')
    ax3.set_xlabel('Number of Nodes', fontsize=12)
    ax3.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax3.set_title('Throughput vs Number of Nodes', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # ===== Plot 4: Hops vs Number of Nodes =====
    ax4 = fig.add_subplot(2, 3, 4)
    for j, speed in enumerate(speeds):
        ax4.plot(node_counts, results['hops'][:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'Speed={speed} m/s')
    ax4.set_xlabel('Number of Nodes', fontsize=12)
    ax4.set_ylabel('Average Hop Count', fontsize=12)
    ax4.set_title('Hop Count vs Number of Nodes', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # ===== Plot 5: PDR Heatmap =====
    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(results['pdr'], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(len(speeds)))
    ax5.set_xticklabels(speeds)
    ax5.set_yticks(range(len(node_counts)))
    ax5.set_yticklabels(node_counts)
    ax5.set_xlabel('Speed (m/s)', fontsize=12)
    ax5.set_ylabel('Number of Nodes', fontsize=12)
    ax5.set_title('PDR Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='PDR')
    # Add text annotations
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            text = ax5.text(j, i, f'{results["pdr"][i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)
    
    # ===== Plot 6: Throughput Heatmap =====
    ax6 = fig.add_subplot(2, 3, 6)
    im2 = ax6.imshow(results['throughput'], cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(len(speeds)))
    ax6.set_xticklabels(speeds)
    ax6.set_yticks(range(len(node_counts)))
    ax6.set_yticklabels(node_counts)
    ax6.set_xlabel('Speed (m/s)', fontsize=12)
    ax6.set_ylabel('Number of Nodes', fontsize=12)
    ax6.set_title('Throughput Heatmap (Mbps)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax6, label='Throughput (Mbps)')
    # Add text annotations
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            text = ax6.text(j, i, f'{results["throughput"][i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    plt.suptitle('MANET Opportunistic Routing: Performance Analysis\n(TGNN + Multi-Agent)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('manet_experiments_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to 'manet_experiments_results.png'")
    
    # Additional: Create separate bar charts
    create_bar_charts(results, node_counts, speeds)


def create_bar_charts(results, node_counts, speeds):
    """Create grouped bar charts for each metric."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(node_counts))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    metrics = [
        ('pdr', 'Packet Delivery Ratio', axes[0, 0]),
        ('delay', 'Average Delay (s)', axes[0, 1]),
        ('throughput', 'Throughput (Mbps)', axes[1, 0]),
        ('hops', 'Average Hop Count', axes[1, 1])
    ]
    
    for metric_key, metric_name, ax in metrics:
        for j, speed in enumerate(speeds):
            offset = (j - len(speeds)/2 + 0.5) * width
            bars = ax.bar(x + offset, results[metric_key][:, j], width, 
                         label=f'{speed} m/s', color=colors[j], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} by Node Count and Speed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(node_counts)
        ax.legend(title='Speed', fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('MANET Performance Metrics: Grouped Bar Charts', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manet_experiments_barcharts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Bar charts saved to 'manet_experiments_barcharts.png'")


def print_summary_table(results, node_counts, speeds):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    
    # PDR Table
    print("\n--- Packet Delivery Ratio (PDR) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{results['pdr'][i, j]:.3f}".center(12), end="")
        print()
    
    # Delay Table
    print("\n--- Average Delay (seconds) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{results['delay'][i, j]:.3f}".center(12), end="")
        print()
    
    # Throughput Table
    print("\n--- Throughput (Mbps) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{results['throughput'][i, j]:.4f}".center(12), end="")
        print()
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    # Run all experiments
    results, node_counts, speeds = run_all_experiments()
    
    # Print summary table
    print_summary_table(results, node_counts, speeds)
    
    # Plot results
    plot_results(results, node_counts, speeds)
