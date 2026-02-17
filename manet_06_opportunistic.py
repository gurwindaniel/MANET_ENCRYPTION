import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# OPPORTUNISTIC ROUTING WITH MULTI-AGENT AND TGNN CANDIDATE FORWARDER TRACKING
# ============================================================================

# === Agent A: Reliability Agent (GNN) ===
class ReliabilityAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = 0.003

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.gnn(x, edge_index)
        out = self.fc(h)
        pdr_score = self.sigmoid(out).squeeze(-1)
        return pdr_score

    def update(self, reward):
        pass

# === Agent B: Candidate Forwarder TGNN (Temporal GNN for Opportunistic Routing) ===
class CandidateForwarderTGNN(nn.Module):
    """
    TGNN-based agent that maintains candidate forwarder list.
    Input features per neighbor:
    - relative_speed: speed difference between nodes
    - link_stability: based on distance change rate
    - contact_duration: predicted contact time
    - direction_alignment: alignment with destination
    - distance_ratio: normalized distance to destination
    - historical_success: past delivery success rate
    - energy_level: remaining energy
    - queue_length: buffer occupancy
    """
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
        self.lr = 0.003
        
        # Candidate history for temporal learning
        self.candidate_history = defaultdict(list)  # neighbor_id -> [(time, score), ...]
        self.contact_history = defaultdict(list)    # neighbor_id -> [(start_time, end_time), ...]

    def forward(self, data):
        """
        Returns forwarding priority scores for each neighbor.
        Higher score = better candidate forwarder.
        """
        x, edge_index = data.x, data.edge_index
        h = self.temporal_gnn(x, edge_index)
        
        # Attention-weighted priority
        attention_weights = self.sigmoid(self.attention(h))
        priority_scores = self.sigmoid(self.fc_priority(h)).squeeze(-1)
        
        # Combine attention and priority
        final_scores = (priority_scores * attention_weights.squeeze(-1))
        return final_scores
    
    def update_contact_history(self, neighbor_id, current_time, in_contact):
        """Track contact periods for link stability estimation."""
        history = self.contact_history[neighbor_id]
        if in_contact:
            if not history or history[-1][1] is not None:
                # Start new contact period
                history.append([current_time, None])
        else:
            if history and history[-1][1] is None:
                # End current contact period
                history[-1][1] = current_time
    
    def get_avg_contact_duration(self, neighbor_id):
        """Get average contact duration with a neighbor."""
        history = self.contact_history[neighbor_id]
        completed = [(end - start) for start, end in history if end is not None]
        return np.mean(completed) if completed else 10.0  # default estimate

    def update(self, reward):
        pass


# === Agent C: Throughput Agent (Contextual Bandit) ===
class ThroughputAgent:
    def __init__(self, max_neighbors):
        self.max_neighbors = max_neighbors
        self.q_values = {}
        self.alpha = 0.3

    def forward(self, context, neighbor_ids):
        throughput_score = 1.0 / (1.0 + context[:, 0])  # inverse queue length
        throughput_score += context[:, 1]  # success ratio
        throughput_score += context[:, 2] * 0.1  # energy factor
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


# === Agent D: Exploration Agent (UCB) ===
class ExplorationAgent:
    def __init__(self, max_neighbors):
        self.max_neighbors = max_neighbors
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


# === Agent E: Enhanced Neighbor Tracking with Mobility Metrics ===
class NeighborTrackingAgent:
    """
    Enhanced neighbor tracking with mobility metrics for opportunistic routing.
    Tracks: position, velocity, relative speed, link stability, contact duration.
    """
    def __init__(self, node, comm_range):
        self.node = node
        self.comm_range = comm_range
        self.neighbors = []
        self.neighbor_metrics = {}  # neighbor_id -> NeighborMetrics
        self.previous_positions = {}  # neighbor_id -> previous position
        
    def update_neighbors(self, all_nodes, current_time, dt=1.0):
        """Update neighbor list and compute mobility metrics."""
        self.neighbors = []
        current_neighbor_set = set()
        
        for other in all_nodes:
            if other.node_id == self.node.node_id:
                continue
                
            dist = np.linalg.norm(self.node.position - other.position)
            
            if dist <= self.comm_range:
                self.neighbors.append(other.node_id)
                current_neighbor_set.add(other.node_id)
                
                # Compute mobility metrics
                metrics = self._compute_neighbor_metrics(other, dist, dt)
                self.neighbor_metrics[other.node_id] = metrics
                
                # Update contact history in TGNN agent
                self.node.agent_b.update_contact_history(other.node_id, current_time, True)
            else:
                # No longer in contact
                if other.node_id in self.neighbor_metrics:
                    self.node.agent_b.update_contact_history(other.node_id, current_time, False)
        
        return self.neighbors
    
    def _compute_neighbor_metrics(self, other_node, dist, dt):
        """Compute detailed mobility metrics for a neighbor."""
        # Relative velocity
        my_velocity = self.node.velocity if hasattr(self.node, 'velocity') else np.zeros(2)
        other_velocity = other_node.velocity if hasattr(other_node, 'velocity') else np.zeros(2)
        relative_velocity = other_velocity - my_velocity
        relative_speed = np.linalg.norm(relative_velocity)
        
        # Link stability (based on approaching/departing)
        direction_to_other = (other_node.position - self.node.position)
        if np.linalg.norm(direction_to_other) > 0:
            direction_to_other = direction_to_other / np.linalg.norm(direction_to_other)
        
        # Positive = approaching, Negative = departing
        approach_rate = -np.dot(relative_velocity, direction_to_other)
        
        # Expected contact duration
        if relative_speed > 0.1:
            remaining_dist = self.comm_range - dist
            if approach_rate > 0:
                # Approaching - contact will last longer
                contact_duration = remaining_dist / relative_speed + dist / relative_speed
            else:
                # Departing - estimate remaining contact time
                contact_duration = max(0, remaining_dist / relative_speed)
        else:
            contact_duration = 100.0  # Long contact if nearly stationary
        
        return {
            'distance': dist,
            'relative_speed': relative_speed,
            'approach_rate': approach_rate,
            'link_stability': 1.0 / (1.0 + abs(approach_rate)),  # Higher if stable
            'contact_duration': contact_duration,
            'direction_to_other': direction_to_other
        }
    
    def get_metrics_for_neighbor(self, neighbor_id):
        """Get metrics for a specific neighbor."""
        return self.neighbor_metrics.get(neighbor_id, None)


# === Data Packet with Store-Carry-Forward Support ===
class DataPacket:
    """Enhanced packet with store-carry-forward and opportunistic routing support."""
    def __init__(self, src, dst, created_time, packet_id=None):
        self.packet_id = packet_id or f"{src}_{dst}_{created_time}"
        self.src = src
        self.dst = dst
        self.created_time = created_time
        self.hops = [src]
        self.delivered = False
        self.delivered_time = None
        
        # Opportunistic routing fields
        self.current_holder = src  # Node currently holding the packet
        self.copies = 1  # Number of copies (for spray-and-wait)
        self.forwarded_to = set()  # Nodes that received this packet (for suppression)
        self.ttl = 50  # Time-to-live in hops


# === Node Class with Opportunistic Routing ===
class Node:
    def __init__(self, node_id, n_neighbors, agent_dims, fusion_weights, comm_range, area_size):
        self.node_id = node_id
        self.n_neighbors = n_neighbors
        
        # Multi-agent system
        self.agent_a = ReliabilityAgent(agent_dims['a_in'], agent_dims['a_hidden'])
        self.agent_b = CandidateForwarderTGNN(agent_dims['b_in'], agent_dims['b_hidden'])  # TGNN for candidates
        self.agent_c = ThroughputAgent(n_neighbors)
        self.agent_d = ExplorationAgent(n_neighbors)
        self.agent_e = NeighborTrackingAgent(self, comm_range)
        
        self.fusion_weights = fusion_weights
        self.position = np.random.rand(2) * area_size
        self.velocity = np.zeros(2)  # Track velocity for relative speed
        self.prev_position = self.position.copy()
        self.area_size = area_size
        self.comm_range = comm_range
        self.mobility_target = self._random_point()
        self.speed = random.uniform(10, 18)
        
        # Store-carry-forward buffer
        self.packet_buffer = []  # Packets being carried
        self.max_buffer_size = 50
        
        # Delivery success history for learning
        self.delivery_history = defaultdict(list)  # neighbor_id -> [success/fail, ...]

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
        
        # Update velocity for metrics
        self.velocity = (self.position - self.prev_position) / dt

    def update_neighbors(self, all_nodes, current_time):
        return self.agent_e.update_neighbors(all_nodes, current_time)
    
    def buffer_packet(self, packet):
        """Add packet to buffer for store-carry-forward."""
        if len(self.packet_buffer) < self.max_buffer_size:
            packet.current_holder = self.node_id
            self.packet_buffer.append(packet)
            return True
        return False
    
    def get_candidate_forwarders(self, all_nodes, packet, current_time, top_k=3):
        """
        Use TGNN and multi-agent fusion to select top-K candidate forwarders.
        Returns list of (neighbor_id, score) tuples sorted by priority.
        """
        neighbor_ids = self.agent_e.neighbors
        
        if not neighbor_ids:
            return []
        
        # Direct delivery check
        if packet.dst in neighbor_ids:
            return [(packet.dst, float('inf'))]
        
        # Filter out nodes already in hop path (loop avoidance)
        valid_neighbors = [nid for nid in neighbor_ids 
                         if nid not in packet.hops and nid not in packet.forwarded_to]
        
        if not valid_neighbors:
            return []
        
        n_neighbors = len(valid_neighbors)
        
        # Build temporal feature matrix for TGNN
        # Features: [rel_speed, link_stability, contact_duration, dist_to_dst, 
        #            direction_alignment, energy, queue_len, historical_success]
        temporal_features = []
        dst_node = all_nodes[packet.dst]
        
        for nid in valid_neighbors:
            metrics = self.agent_e.get_metrics_for_neighbor(nid)
            neighbor_node = all_nodes[nid]
            
            if metrics:
                rel_speed = metrics['relative_speed']
                link_stability = metrics['link_stability']
                contact_duration = metrics['contact_duration']
                
                # Distance to destination (normalized)
                dist_to_dst = np.linalg.norm(neighbor_node.position - dst_node.position)
                my_dist_to_dst = np.linalg.norm(self.position - dst_node.position)
                dist_ratio = dist_to_dst / (my_dist_to_dst + 1e-6)  # <1 means closer
                
                # Direction alignment with destination
                dir_to_dst = dst_node.position - neighbor_node.position
                if np.linalg.norm(dir_to_dst) > 0:
                    dir_to_dst = dir_to_dst / np.linalg.norm(dir_to_dst)
                neighbor_vel = neighbor_node.velocity
                if np.linalg.norm(neighbor_vel) > 0:
                    neighbor_vel_norm = neighbor_vel / np.linalg.norm(neighbor_vel)
                    direction_alignment = np.dot(neighbor_vel_norm, dir_to_dst)
                else:
                    direction_alignment = 0.0
                
                # Historical success rate
                hist = self.delivery_history.get(nid, [])
                success_rate = np.mean(hist) if hist else 0.5
                
                # Energy and queue (simulated)
                energy = random.uniform(0.5, 1.0)
                queue_len = len(neighbor_node.packet_buffer) / neighbor_node.max_buffer_size
                
                temporal_features.append([
                    rel_speed / 20.0,  # Normalized
                    link_stability,
                    min(contact_duration, 100) / 100.0,  # Normalized
                    1.0 - min(dist_ratio, 2.0) / 2.0,  # Inverted: closer is better
                    (direction_alignment + 1) / 2.0,  # Normalized to [0,1]
                    energy,
                    1.0 - queue_len,  # Inverted: less queue is better
                    success_rate
                ])
            else:
                temporal_features.append([0.5] * 8)
        
        temporal_features = np.array(temporal_features, dtype=np.float32)
        
        # Create graph data for agents
        edge_index = self._create_neighbor_graph_edges(n_neighbors)
        
        neighbor_graph = Data(
            x=torch.tensor(temporal_features, dtype=torch.float32),
            edge_index=edge_index
        )
        
        # Get scores from all agents
        with torch.no_grad():
            pdr_scores = self.agent_a(neighbor_graph).cpu().numpy()
            tgnn_scores = self.agent_b(neighbor_graph).cpu().numpy()  # TGNN candidate scores
        
        # Context for throughput agent
        context_data = temporal_features[:, [6, 7, 5]]  # queue, success, energy
        throughput_scores = self.agent_c.forward(context_data, valid_neighbors)
        exploration_bonuses = self.agent_d.forward(valid_neighbors)
        
        # Align arrays
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
        
        # Multi-agent fusion
        alpha = self.fusion_weights['alpha']
        beta = self.fusion_weights['beta']  # Now for TGNN candidate score
        gamma = self.fusion_weights['gamma']
        delta = self.fusion_weights['delta']
        
        # === DELAY OPTIMIZATION: Strong preference for progress toward destination ===
        distance_progress = temporal_features[:, 3]  # Closer to destination = higher score
        link_stability = temporal_features[:, 1]     # More stable link = faster transfer
        contact_duration = temporal_features[:, 2]   # Longer contact = more packets
        low_queue = temporal_features[:, 6]          # Less congested = faster processing
        
        final_scores = (
            alpha * pdr_scores
            + beta * tgnn_scores  # TGNN-based candidate priority
            + gamma * throughput_scores
            + delta * exploration_bonuses
            + 4.0 * distance_progress  # Very strong preference for progress (was 2.0)
            + 1.5 * link_stability     # Prefer stable links
            + 1.0 * contact_duration   # Prefer longer contacts
            + 0.5 * low_queue          # Prefer less congested nodes
        )
        
        # Sort by score and return top-K candidates
        candidates = [(valid_neighbors[i], final_scores[i]) for i in range(n_neighbors)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def _create_neighbor_graph_edges(self, n_neighbors):
        """Create a simple fully-connected graph for neighbors."""
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
        """Record delivery success/failure for learning."""
        self.delivery_history[neighbor_id].append(1.0 if success else 0.0)
        # Keep only recent history
        if len(self.delivery_history[neighbor_id]) > 100:
            self.delivery_history[neighbor_id] = self.delivery_history[neighbor_id][-100:]


# === Opportunistic Routing Simulation ===
def update_node_positions(nodes, dt=1.0):
    for node in nodes:
        node.move(dt=dt)


def simulate_opportunistic(nodes, num_steps, area_size):
    """
    Opportunistic routing simulation with:
    - TGNN-based candidate forwarder selection
    - Store-carry-forward mechanism
    - Multi-copy forwarding with duplicate suppression
    - Multiple forwarding rounds per step for reduced delay
    - Batch forwarding for improved throughput
    """
    metrics = {
        'pdr': [],
        'throughput': [],
        'delay': [],
        'delivered_count': [],
        'avg_delay': [],
        'throughput_mbps': [],
        'buffer_occupancy': [],
        'forwarding_opportunities': []
    }
    
    packets = []
    delivered_packets = []
    delivered_packet_ids = set()  # For duplicate suppression
    fed_interval = 40
    
    # === THROUGHPUT OPTIMIZATION ===
    PACKET_SIZE_BYTES = 8192  # Increased from 4096 to 8KB
    PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8
    FORWARDING_ROUNDS_PER_STEP = 3  # Multiple forwarding rounds per time step
    MAX_PACKETS_PER_CONTACT = 5  # Batch forwarding: multiple packets per opportunity
    
    packet_counter = 0
    
    for t in range(num_steps):
        # Move nodes
        update_node_positions(nodes)
        
        # Update neighbors with current time
        for node in nodes:
            node.update_neighbors(nodes, t)
        
        # Generate packets (increased rate for higher throughput)
        if t % 3 == 0:  # More frequent packet generation
            for _ in range(15):  # More packets per interval
                src = random.choice(nodes)
                dst = random.choice([n for n in nodes if n.node_id != src.node_id])
                packet = DataPacket(src.node_id, dst.node_id, t, packet_id=packet_counter)
                packet_counter += 1
                packets.append(packet)
                src.buffer_packet(packet)
        
        # === Opportunistic Forwarding with Multiple Rounds ===
        forwarding_opportunities = 0
        
        # Multiple forwarding rounds per time step for reduced delay
        for forwarding_round in range(FORWARDING_ROUNDS_PER_STEP):
            for node in nodes:
                packets_to_remove = []
                packets_forwarded_this_contact = 0
                
                # Sort buffer by urgency (older packets first, closer destinations priority)
                sorted_buffer = sorted(
                    node.packet_buffer,
                    key=lambda p: (t - p.created_time, -len(p.hops)),
                    reverse=True
                )
                
                for packet in sorted_buffer:
                    # Batch limit per contact
                    if packets_forwarded_this_contact >= MAX_PACKETS_PER_CONTACT:
                        break
                    
                    if packet.delivered or packet.packet_id in delivered_packet_ids:
                        packets_to_remove.append(packet)
                        continue
                    
                    if packet.ttl <= 0:
                        packets_to_remove.append(packet)
                        continue
                    
                    # Get candidate forwarders using TGNN and multi-agent
                    candidates = node.get_candidate_forwarders(nodes, packet, t, top_k=5)  # More candidates
                    
                    if not candidates:
                        continue
                    
                    forwarding_opportunities += 1
                    
                    # Opportunistic forwarding: try candidates in priority order
                    forwarded = False
                    for candidate_id, score in candidates:
                        candidate_node = nodes[candidate_id]
                        
                        # Check if direct delivery
                        if candidate_id == packet.dst:
                            packet.hops.append(candidate_id)
                            packet.delivered = True
                            packet.delivered_time = t
                            delivered_packets.append(packet)
                            delivered_packet_ids.add(packet.packet_id)
                            packets_to_remove.append(packet)
                            
                            # High reward for successful delivery
                            delay = t - packet.created_time
                            delay_bonus = max(0, 1.0 - delay * 0.1)  # Bonus for fast delivery
                            rewards = {
                                'a': 1.0 + delay_bonus,
                                'b': 1.0 + delay_bonus,
                                'c': 1.0 + delay_bonus,
                                'd': 1.0 + delay_bonus
                            }
                            node.update_agents(rewards, candidate_id)
                            node.record_delivery_result(candidate_id, True)
                            forwarded = True
                            packets_forwarded_this_contact += 1
                            break
                        
                        # Forward to candidate (opportunistic)
                        if candidate_node.buffer_packet(packet):
                            packet.hops.append(candidate_id)
                            packet.forwarded_to.add(candidate_id)
                            packet.ttl -= 1
                            
                            # Calculate reward based on progress toward destination
                            dst_node = nodes[packet.dst]
                            old_dist = np.linalg.norm(node.position - dst_node.position)
                            new_dist = np.linalg.norm(candidate_node.position - dst_node.position)
                            progress_reward = max(0, (old_dist - new_dist) / old_dist) if old_dist > 0 else 0
                            
                            # Higher rewards for significant progress
                            rewards = {
                                'a': 0.4 + progress_reward * 0.6,
                                'b': 0.4 + progress_reward * 0.6,
                                'c': 0.4 + progress_reward * 0.6,
                                'd': 0.4 + progress_reward * 0.6
                            }
                            node.update_agents(rewards, candidate_id)
                            
                            packets_to_remove.append(packet)
                            forwarded = True
                            packets_forwarded_this_contact += 1
                            break
                
                # Remove forwarded/delivered packets from buffer
                for pkt in packets_to_remove:
                    if pkt in node.packet_buffer:
                        node.packet_buffer.remove(pkt)
        
        # Federated learning
        if t > 0 and t % fed_interval == 0:
            federated_learning(nodes)
        
        # === Metrics Collection ===
        delivered = [p for p in packets if p.delivered]
        pdr = len(delivered) / len(packets) if packets else 0
        metrics['pdr'].append(pdr)
        metrics['delivered_count'].append(len(delivered))
        
        if delivered:
            current_avg_delay = np.mean([p.delivered_time - p.created_time for p in delivered])
        else:
            current_avg_delay = 0.0
        metrics['avg_delay'].append(current_avg_delay)
        
        total_bits = len(delivered) * PACKET_SIZE_BITS
        current_throughput = total_bits / (t + 1) / 1e6 if (t + 1) > 0 else 0.0
        metrics['throughput_mbps'].append(current_throughput)
        
        # Buffer occupancy
        total_buffer = sum(len(n.packet_buffer) for n in nodes)
        max_buffer = sum(n.max_buffer_size for n in nodes)
        metrics['buffer_occupancy'].append(total_buffer / max_buffer if max_buffer > 0 else 0)
        
        metrics['forwarding_opportunities'].append(forwarding_opportunities)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPPORTUNISTIC ROUTING SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Delivered: {len(delivered_packets)}/{len(packets)}")
    print(f"Avg PDR: {np.mean(metrics['pdr']) if metrics['pdr'] else 0:.3f}")
    
    if delivered_packets:
        avg_delay = np.mean([p.delivered_time - p.created_time for p in delivered_packets])
        avg_hops = np.mean([len(p.hops) - 1 for p in delivered_packets])
    else:
        avg_delay = 0.0
        avg_hops = 0.0
    
    total_bits = len(delivered_packets) * PACKET_SIZE_BITS
    throughput_mbps = total_bits / num_steps / 1e6 if num_steps > 0 else 0.0
    
    print(f"Avg Delay (s): {avg_delay:.3f}")
    print(f"Avg Hops: {avg_hops:.2f}")
    print(f"Throughput (Mbps): {throughput_mbps:.6f}")
    print(f"Avg Buffer Occupancy: {np.mean(metrics['buffer_occupancy']):.2%}")
    print(f"{'='*60}\n")
    
    plot_opportunistic_metrics(metrics, num_steps)


def plot_opportunistic_metrics(metrics, num_steps):
    """Plot metrics for opportunistic routing simulation."""
    time_steps = list(range(num_steps))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Opportunistic MANET Routing with TGNN Candidate Selection', fontsize=14, fontweight='bold')
    
    # Plot 1: Delivered Packets
    ax1 = axes[0, 0]
    ax1.plot(time_steps, metrics['delivered_count'], color='green', linewidth=1.5)
    ax1.set_xlabel('Time Step (s)')
    ax1.set_ylabel('Delivered Packets')
    ax1.set_title('Delivered Packets Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.fill_between(time_steps, metrics['delivered_count'], alpha=0.3, color='green')
    
    # Plot 2: PDR
    ax2 = axes[0, 1]
    ax2.plot(time_steps, metrics['pdr'], color='blue', linewidth=1.5)
    ax2.set_xlabel('Time Step (s)')
    ax2.set_ylabel('Packet Delivery Ratio')
    ax2.set_title('PDR Over Time')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.fill_between(time_steps, metrics['pdr'], alpha=0.3, color='blue')
    
    # Plot 3: Average Delay
    ax3 = axes[0, 2]
    ax3.plot(time_steps, metrics['avg_delay'], color='red', linewidth=1.5)
    ax3.set_xlabel('Time Step (s)')
    ax3.set_ylabel('Delay (seconds)')
    ax3.set_title('Average Delay Over Time')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.fill_between(time_steps, metrics['avg_delay'], alpha=0.3, color='red')
    
    # Plot 4: Throughput
    ax4 = axes[1, 0]
    ax4.plot(time_steps, metrics['throughput_mbps'], color='purple', linewidth=1.5)
    ax4.set_xlabel('Time Step (s)')
    ax4.set_ylabel('Throughput (Mbps)')
    ax4.set_title('Throughput Over Time')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.fill_between(time_steps, metrics['throughput_mbps'], alpha=0.3, color='purple')
    
    # Plot 5: Buffer Occupancy
    ax5 = axes[1, 1]
    ax5.plot(time_steps, [b * 100 for b in metrics['buffer_occupancy']], color='orange', linewidth=1.5)
    ax5.set_xlabel('Time Step (s)')
    ax5.set_ylabel('Buffer Occupancy (%)')
    ax5.set_title('Network Buffer Utilization')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.fill_between(time_steps, [b * 100 for b in metrics['buffer_occupancy']], alpha=0.3, color='orange')
    
    # Plot 6: Forwarding Opportunities
    ax6 = axes[1, 2]
    ax6.plot(time_steps, metrics['forwarding_opportunities'], color='teal', linewidth=1.5)
    ax6.set_xlabel('Time Step (s)')
    ax6.set_ylabel('Opportunities')
    ax6.set_title('Forwarding Opportunities per Step')
    ax6.grid(True, linestyle='--', alpha=0.7)
    ax6.fill_between(time_steps, metrics['forwarding_opportunities'], alpha=0.3, color='teal')
    
    plt.tight_layout()
    plt.savefig('manet_opportunistic_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Metrics plot saved as 'manet_opportunistic_metrics.png'")


# === Federated Learning ===
def federated_average(models):
    avg_state = {}
    n = len(models)
    for k in models[0].state_dict().keys():
        avg_state[k] = sum([m.state_dict()[k].float() for m in models]) / n
    for m in models:
        m.load_state_dict(avg_state)


def federated_learning(nodes):
    # Aggregate neural network agents
    reliability_agents = [n.agent_a for n in nodes]
    tgnn_agents = [n.agent_b for n in nodes]
    federated_average(reliability_agents)
    federated_average(tgnn_agents)
    
    # Aggregate ThroughputAgent q_values
    all_keys = set()
    for node in nodes:
        all_keys.update(node.agent_c.q_values.keys())
    
    if all_keys:
        avg_q_values = {}
        for key in all_keys:
            values = [n.agent_c.q_values.get(key, 0.0) for n in nodes]
            avg_q_values[key] = np.mean(values)
        for node in nodes:
            node.agent_c.q_values = avg_q_values.copy()
    
    # Aggregate ExplorationAgent
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
        for node in nodes:
            node.agent_d.values = avg_values.copy()
            node.agent_d.counts = avg_counts.copy()


# === Main ===
if __name__ == "__main__":
    print("Initializing Opportunistic MANET Routing Simulation...")
    print("Using TGNN for candidate forwarder selection with mobility metrics")
    print("Optimized for HIGH THROUGHPUT and LOW DELAY")
    print("-" * 60)
    
    # === OPTIMIZED PARAMETERS ===
    n_nodes = 100
    n_neighbors = 15  # Increased for more routing options
    area_size = 800   # Smaller area = higher density = more contacts
    comm_range = 300  # Increased range = more neighbors = faster forwarding
    
    agent_dims = {
        'a_in': 8, 'a_hidden': 64,   # Larger hidden layer for better learning
        'b_in': 8, 'b_hidden': 64
    }
    
    # Fusion weights optimized for low delay
    # High weight on distance progress for faster delivery
    fusion_weights = {
        'alpha': 1.5,   # Reliability
        'beta': 2.5,    # TGNN candidate score
        'gamma': 2.0,   # Throughput
        'delta': 0.3    # Lower exploration (exploit good paths)
    }
    
    nodes = [
        Node(i, n_neighbors, agent_dims, fusion_weights, comm_range, area_size)
        for i in range(n_nodes)
    ]
    
    # Increase node speed for faster contacts
    for node in nodes:
        node.speed = random.uniform(15, 25)  # Faster movement
        node.max_buffer_size = 100  # Larger buffer for more throughput
    
    simulate_opportunistic(nodes, num_steps=400, area_size=area_size)
