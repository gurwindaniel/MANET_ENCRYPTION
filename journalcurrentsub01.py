import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
import torch.nn as nn
import torch.nn.functional as F
# Add PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import math  # Already imported, but needed for log in UCB
from collections import deque # Import deque
import seaborn as sns

# Detect if GPU is available
device = torch.device('cpu') # Changed to CPU to avoid OutOfMemoryError
print(f'Using device: {device}')

class GNNModel(nn.Module):
    """Further tuned GCN-based GNN for neighbor scoring with residual connections and normalization."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        # Removed conv4, ln4, conv5, ln5
        self.conv6 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(0.18)  # Reduced dropout for better retention

    def forward(self, x, edge_index):
        x1 = F.gelu(self.ln1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        x2 = F.gelu(self.ln2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        x3 = F.gelu(self.ln3(self.conv3(x2 + x1, edge_index))) # Residual connection
        x3 = self.dropout(x3)
        # Removed x4, x5
        out = self.conv6(x3, edge_index)
        return out.squeeze(-1)

class OnlineGNN:
    """Online GNN agent for neighbor discovery and scoring using real GNN."""
    def __init__(self, node, all_nodes, input_dim=5, hidden_dim=128, neighbor_radius=350):
        self.node = node
        self.all_nodes = all_nodes
        self.neighbor_radius = neighbor_radius
        self.neighbors = set()
        self.neighbor_features = {}  # node_id: feature vector
        self.gnn = GNNModel(input_dim, hidden_dim).to(device)  # Move model to device
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.002)
        self.last_edge_index = None
        self.last_x = None
        self.mab_counts = {}   # neighbor_id: count of selections
        self.mab_rewards = {}  # neighbor_id: total reward
        self.total_mab_selections = 0  # For UCB

    def update_neighbors(self):
        self.neighbors.clear()
        self.neighbor_features.clear()
        node_pos = np.array([self.node.x, self.node.y, self.node.z])
        for other in self.all_nodes:
            if other.node_id == self.node.node_id:
                continue
            other_pos = np.array([other.x, other.y, other.z])
            dist = np.linalg.norm(node_pos - other_pos)
            if dist <= self.neighbor_radius:
                self.neighbors.add(other.node_id)
                feat = np.array([
                    other.x, other.y, other.z,
                    getattr(other, 'energy', 100.0),
                    dist
                ], dtype=np.float32)
                self.neighbor_features[other.node_id] = feat

    def get_neighbors(self):
        return self.neighbors

    def _build_graph(self):
        """Builds a local graph for message passing."""
        # Node 0: self, Node 1...N: neighbors
        node_ids = [self.node.node_id] + list(self.neighbor_features.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        x = [np.array([self.node.x, self.node.y, self.node.z, getattr(self.node, 'energy', 100.0), 0.0], dtype=np.float32)]
        x += [self.neighbor_features[nid] for nid in self.neighbor_features]
        x = np.array(x, dtype=np.float32)  # Convert list to numpy array first
        x = torch.tensor(x, dtype=torch.float32).to(device) # Move tensor to device
        # Fully connect self to neighbors (undirected)
        edge_index = []
        for nid in self.neighbor_features:
            i = 0  # self
            j = id_to_idx[nid]
            edge_index.append([i, j])
            edge_index.append([j, i])
        if not edge_index:
            edge_index = torch.empty((2,0), dtype=torch.long).to(device) # Move tensor to device
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device) # Move tensor to device
        return x, edge_index, id_to_idx

    def compute_embeddings(self):
        """Compute scores for each neighbor using GNN with message passing."""
        if not self.neighbor_features:
            return {}
        x, edge_index, id_to_idx = self._build_graph()
        self.last_x = x
        self.last_edge_index = edge_index
        with torch.no_grad():
            scores = self.gnn(x, edge_index)
        # Optionally: fallback to random scores if no GNN output
        return {nid: float(scores[id_to_idx[nid]]) for nid in self.neighbor_features}

    def online_update(self, feedback):
        """
        Online update using real feedback.
        feedback: dict {neighbor_id: reward/score}, e.g., 1 for successful delivery, 0 for fail.
        """
        if not self.neighbor_features or not feedback:
            return
        if self.last_x is None or self.last_edge_index is None:
            return
        x = self.last_x
        edge_index = self.last_edge_index
        id_to_idx = {nid: i for i, nid in enumerate([self.node.node_id] + list(self.neighbor_features.keys()))}
        targets = torch.zeros(x.size(0)).to(device) # Move tensor to device
        for nid, val in feedback.items():
            idx = id_to_idx.get(nid, None)
            if idx is not None and idx < len(targets):
                targets[idx] = val
        pred = self.gnn(x, edge_index)
        loss = F.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def mab_select_forwarder(self, candidates, dst_node=None):
        """UCB with much higher exploration and reward shaping. Only consider progress.
        --- TGNN integration: bias UCB using normalized predicted remaining link lifetime ---
        """
        self.total_mab_selections += 1
        ucb_scores = {}
        # --- Compute TGNN-based normalization parameters ---
        tgnn_lifetimes = []
        for nid in candidates:
            pred_lifetime = self.node.tgnn_predict_link_lifetime(nid)
            if pred_lifetime is not None and np.isfinite(pred_lifetime):
                tgnn_lifetimes.append(pred_lifetime)
        # Normalize predicted lifetimes for this node (min-max, avoid division by zero)
        min_life = min(tgnn_lifetimes) if tgnn_lifetimes else 0.0
        max_life = max(tgnn_lifetimes) if tgnn_lifetimes else 1.0
        norm = lambda x: (x - min_life) / (max_life - min_life + 1e-6)
        for nid in candidates:
            count = self.mab_counts.get(nid, 0)
            reward = self.mab_rewards.get(nid, 0)
            count = count + 1e-2
            reward = reward + 0.1
            avg_reward = reward / count
            ucb = avg_reward + math.sqrt(128 * math.log(self.total_mab_selections + 1) / count)
            ucb += random.uniform(0, 0.01)
            # --- TGNN: bias UCB by predicted remaining link lifetime (normalized, small weight) ---
            pred_lifetime = self.node.tgnn_predict_link_lifetime(nid)
            if pred_lifetime is not None and np.isfinite(pred_lifetime):
                ucb += 0.1 * norm(pred_lifetime)  # 0.1 is a small bias factor; tune as needed
            # Only allow progress neighbors if dst_node is given
            if dst_node is not None:
                my_pos = np.array([self.node.x, self.node.y, self.node.z])
                dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
                neighbor = next((n for n in self.all_nodes if n.node_id == nid), None)
                if neighbor is not None:
                    neighbor_pos = np.array([neighbor.x, neighbor.y, neighbor.z])
                    if np.linalg.norm(neighbor_pos - dst_pos) >= np.linalg.norm(my_pos - dst_pos):
                        ucb -= 1000
            ucb_scores[nid] = ucb
        # Prioritize the closest neighbor to the destination among the best UCBs
        if dst_node is not None and ucb_scores:
            dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
            min_dist = float('inf')
            best_nid = None
            for nid in sorted(ucb_scores, key=ucb_scores.get, reverse=True):
                neighbor = next((n for n in self.all_nodes if n.node_id == nid), None)
                if neighbor is not None:
                    neighbor_pos = np.array([neighbor.x, neighbor.y, neighbor.z])
                    dist = np.linalg.norm(neighbor_pos - dst_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_nid = nid
            return best_nid
        return max(ucb_scores, key=ucb_scores.get)

    def mab_update(self, neighbor_id, reward, src_node=None, dst_node=None, forwarder_node=None):
        # Reward shaping: much larger bonus for progress
        shaped_reward = reward
        if src_node is not None and dst_node is not None and forwarder_node is not None:
            src_pos = np.array([src_node.x, src_node.y, src_node.z])
            dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
            fwd_pos = np.array([forwarder_node.x, forwarder_node.y, forwarder_node.z])
            src_dist = np.linalg.norm(src_pos - dst_pos)
            fwd_dist = np.linalg.norm(fwd_pos - dst_pos)
            if fwd_dist < src_dist:
                shaped_reward += 20.0  # Even larger partial reward for progress
        self.mab_counts[neighbor_id] = self.mab_counts.get(neighbor_id, 0) + 1
        self.mab_rewards[neighbor_id] = self.mab_rewards.get(neighbor_id, 0) + shaped_reward

class TGNNLinkLifetime(nn.Module):
    """
    Lightweight Temporal GNN for predicting remaining link lifetime.
    Input: sequence of link features (relative pos, velocity, direction, energy, link age, distance trend)
    Output: predicted remaining link lifetime (scalar)
    """
    def __init__(self, feature_dim=8, hidden_dim=64, window=5):
        super().__init__()
        self.gcn1 = GCNConv(feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.window = window

    def forward(self, x_seq, edge_index_seq):
        # x_seq: (window, nodes, feature_dim)
        # edge_index_seq: list of edge_index for each window step
        gcn_outs = []
        for t in range(self.window):
            x_t = x_seq[t]  # (nodes, feature_dim)
            edge_index_t = edge_index_seq[t]
            # --- Fix: If only one node, skip GCN and use input directly to avoid scatter error ---
            if x_t.size(0) == 1 or edge_index_t.numel() == 0:
                gcn2_out = x_t
            else:
                gcn1_out = torch.relu(self.gcn1(x_t, edge_index_t))
                gcn2_out = torch.relu(self.gcn2(gcn1_out, edge_index_t))
            gcn_outs.append(gcn2_out.unsqueeze(0))  # (1, nodes, hidden_dim) or (1, nodes, feature_dim)
        gcn_outs = torch.cat(gcn_outs, dim=0)  # (window, nodes, hidden_dim/feature_dim)
        # If input was passed through, project to hidden_dim for GRU
        if gcn_outs.shape[-1] != self.gru.input_size:
            gcn_outs = gcn_outs if gcn_outs.shape[-1] == self.gru.input_size else torch.cat([gcn_outs]*self.gru.input_size, dim=-1)[..., :self.gru.input_size]
        gcn_outs = gcn_outs.permute(1, 0, 2)  # (nodes, window, hidden_dim)
        out, _ = self.gru(gcn_outs)  # (nodes, window, hidden_dim)
        last_out = out[:, -1, :]  # (nodes, hidden_dim)
        pred = self.fc(last_out)  # (nodes, 1)
        return pred.squeeze(-1)   # (nodes,)

# --- Patch Node to include TGNN state encoder ---
class Node:
    def __init__(self, node_id, address, mac_address):
        self.node_id = node_id
        self.address = address
        self.mac = mac_address
        self.x = 0
        self.y = 0
        self.z = 0
        self.energy = 100.0  # Initial energy in Joules
        self.gnn = None  # Will be set after all nodes are created
        self.forwarding_queue = []  # Queue for multi-hop forwarding
        self.position_history = deque(maxlen=5) # Store own recent positions
        self.link_state_tracker = {} # Store state information for each neighbor
        self.tgnn = TGNNLinkLifetime(feature_dim=8, hidden_dim=64, window=5).to(device)
        self.tgnn_optimizer = torch.optim.Adam(self.tgnn.parameters(), lr=0.001)
        self.tgnn_history = {}  # neighbor_id: deque of feature vectors (window)
        self.tgnn_targets = []  # (features, edge_index_seq, target_remaining_lifetime)
        self.tgnn_last_break_time = {}  # neighbor_id: last break time for debugging/analysis

    def __repr__(self):
        return f"Node({self.node_id})"

    def update_position_history(self, current_time):
        self.position_history.append((current_time, self.x, self.y, self.z))

    def _calculate_relative_features(self, neighbor_node, current_time):
        my_pos = np.array([self.x, self.y, self.z])
        neighbor_pos = np.array([neighbor_node.x, neighbor_node.y, neighbor_node.z])  # <-- Fix typo here
        relative_pos = neighbor_pos - my_pos
        relative_dist = np.linalg.norm(relative_pos)
        return (relative_pos[0], relative_pos[1], relative_pos[2], relative_dist)

    def update_link_state_tracker(self, current_time, current_neighbors_ids, all_nodes):
        # Convert all_nodes list to a dictionary for efficient lookup
        all_nodes_dict = {node.node_id: node for node in all_nodes}

        # Update existing links and add new links
        for neighbor_id in current_neighbors_ids:
            neighbor_node = all_nodes_dict.get(neighbor_id)
            if neighbor_node is None:
                continue
            relative_x, relative_y, relative_z, relative_dist = self._calculate_relative_features(neighbor_node, current_time)
            vx, vy, vz = self.calculate_relative_velocity(neighbor_id)
            direction = np.arctan2(vy, vx) if vx or vy else 0.0
            energy = getattr(neighbor_node, 'energy', 100.0)
            link_age = current_time - self.link_state_tracker.get(neighbor_id, {}).get('start_time', current_time)
            dist_trend = self.calculate_distance_trend(neighbor_id)
            # --- Compose TGNN feature vector ---
            tgnn_feat = np.array([
                relative_x, relative_y, relative_z,
                vx, vy, vz,
                energy,
                link_age,
                dist_trend
            ], dtype=np.float32)
            # Maintain sliding window for TGNN
            if neighbor_id not in self.tgnn_history:
                self.tgnn_history[neighbor_id] = deque(maxlen=5)
            self.tgnn_history[neighbor_id].append(tgnn_feat)
            if neighbor_id not in self.link_state_tracker:
                # New link
                self.link_state_tracker[neighbor_id] = {
                    'start_time': current_time,
                    'feature_history': deque(maxlen=5),
                    'last_seen_timestamp': current_time,
                    'last_relative_pos': (relative_x, relative_y, relative_z, relative_dist)
                }
                self.link_state_tracker[neighbor_id]['feature_history'].append((current_time, relative_x, relative_y, relative_z, relative_dist))
            else:
                # Existing link
                self.link_state_tracker[neighbor_id]['feature_history'].append((current_time, relative_x, relative_y, relative_z, relative_dist))
                self.link_state_tracker[neighbor_id]['last_seen_timestamp'] = current_time
                self.link_state_tracker[neighbor_id]['last_relative_pos'] = (relative_x, relative_y, relative_z, relative_dist)

        # --- Handle broken links and generate TGNN targets for remaining lifetime ---
        for tracked_neighbor_id in list(self.link_state_tracker.keys()):
            if tracked_neighbor_id not in current_neighbors_ids:
                link_info = self.link_state_tracker[tracked_neighbor_id]
                link_break_time = current_time
                # For each observation in the sliding window, generate a training sample with remaining lifetime
                tgnn_hist = self.tgnn_history.get(tracked_neighbor_id, None)
                if tgnn_hist and len(tgnn_hist) == 5:
                    # For each time step in the window, compute remaining lifetime
                    # We use the time of each observation in the window
                    # For simplicity, assume uniform time steps and use the last 5 observations
                    obs_times = []
                    for f in tgnn_hist:
                        # The first 3 are relative pos, then vx, vy, vz, then energy, link_age, dist_trend
                        # We need to get the time of the observation; since it's not in the feature, we can reconstruct from link_info['feature_history']
                        # We'll match by order (sliding window)
                        pass
                    # Instead, use the last 5 entries in link_info['feature_history'] for times
                    feature_hist = list(link_info['feature_history'])[-5:]
                    for idx, (obs_time, *_rest) in enumerate(feature_hist):
                        # Compose the feature vector for this observation
                        feat = tgnn_hist[idx]
                        remaining_lifetime = link_break_time - obs_time  # This is the correct target
                        # Build dummy edge_index_seq (fully connected for 2 nodes)
                        edge_index_seq = [torch.tensor([[0,1],[1,0]], dtype=torch.long).to(device) for _ in range(5)]
                        # For each sample, build a window ending at this observation (if enough history)
                        # For simplicity, only use the last window (full 5)
                        if idx == 4:
                            x_seq = torch.stack([torch.tensor(f, dtype=torch.float32).unsqueeze(0) for f in tgnn_hist])  # (window, 1, feat)
                            self.tgnn_targets.append((x_seq, edge_index_seq, torch.tensor([remaining_lifetime], dtype=torch.float32).to(device)))
                # Store last break time for analysis/debug
                self.tgnn_last_break_time[tracked_neighbor_id] = link_break_time
                del self.link_state_tracker[tracked_neighbor_id]
                if tracked_neighbor_id in self.tgnn_history:
                    del self.tgnn_history[tracked_neighbor_id]

    def calculate_relative_velocity(self, neighbor_id):
        if neighbor_id not in self.link_state_tracker or len(self.position_history) < 2:
            return (0.0, 0.0, 0.0) # Default to zero velocity if not enough data

        # Get own last two positions
        t_self_1, x_self_1, y_self_1, z_self_1 = self.position_history[-1]
        t_self_0, x_self_0, y_self_0, z_self_0 = self.position_history[-2]

        # Get neighbor's last two relative positions
        # We need at least two entries in feature_history to calculate relative velocity
        feature_history = self.link_state_tracker[neighbor_id]['feature_history']
        if len(feature_history) < 2:
            return (0.0, 0.0, 0.0) # Default to zero velocity if not enough data

        # Current relative position
        t_rel_1, rel_x_1, rel_y_1, rel_z_1, _ = feature_history[-1]
        # Previous relative position (or 2nd to last if more exist)
        t_rel_0, rel_x_0, rel_y_0, rel_z_0, _ = feature_history[-2]

        # Approximate current absolute position of neighbor (from self perspective)
        neighbor_abs_x_1 = x_self_1 + rel_x_1
        neighbor_abs_y_1 = y_self_1 + rel_y_1
        neighbor_abs_z_1 = z_self_1 + rel_z_1

        # Approximate previous absolute position of neighbor (from self perspective)
        # Note: This assumes t_self_0 == t_rel_0 approximately. Ideally, we would match timestamps.
        neighbor_abs_x_0 = x_self_0 + rel_x_0
        neighbor_abs_y_0 = y_self_0 + rel_y_0
        neighbor_abs_z_0 = z_self_0 + rel_z_0

        dt = t_self_1 - t_self_0 # Time difference should be consistent

        if dt == 0:
            return (0.0, 0.0, 0.0)

        vx = (neighbor_abs_x_1 - neighbor_abs_x_0) / dt
        vy = (neighbor_abs_y_1 - neighbor_abs_y_0) / dt
        vz = (neighbor_abs_z_1 - neighbor_abs_z_0) / dt

        return (vx, vy, vz)

    def calculate_distance_trend(self, neighbor_id):
        if neighbor_id not in self.link_state_tracker:
            return 0.0 # Default to no trend

        feature_history = self.link_state_tracker[neighbor_id]['feature_history']
        if len(feature_history) < 2:
            return 0.0 # Need at least two points to calculate trend

        # Get the two most recent distance readings
        last_dist_entry = feature_history[-1]
        second_last_dist_entry = feature_history[-2]

        current_time = last_dist_entry[0]
        current_dist = last_dist_entry[4] # Index 4 is relative_dist

        prev_time = second_last_dist_entry[0]
        prev_dist = second_last_dist_entry[4]

        dt = current_time - prev_time

        if dt == 0:
            return 0.0 # Avoid division by zero, no change implies no trend

        distance_change = current_dist - prev_dist
        distance_trend = distance_change / dt

        return distance_trend

    def construct_frame(self):
        # Multicast IPv4 address (example: 224.1.1.1)
        multicast_addr = "224.1.1.1"
        # Broadcast MAC address
        broadcast_mac = "ff:ff:ff:ff:ff:ff"
        # UDP ports
        src_port = random.randint(1025, 65535)
        dst_port = 1023
        # Create Hello payload
        hello = Hello(self.node_id, self.x, self.y, self.z)
        udp_payload = str(hello).encode()
        udp_length = 8 + len(udp_payload)
        udp_header = UDPHeader(src_port, dst_port, length=udp_length)
        ipv4_total_length = 20 + udp_length
        ipv4_header = IPv4Header(self.address, multicast_addr)
        ipv4_header.total_length = ipv4_total_length
        datalink = DataLink80211(self.mac, broadcast_mac)
        return {
            "datalink": datalink,
            "ipv4": ipv4_header,
            "udp": udp_header,
            "payload": hello
        }

    def receive_frame(self, frame):
        # Extract Hello payload and update GNN
        hello = frame["payload"]
        node_pos = np.array([self.x, self.y, self.z])
        hello_pos = np.array([hello.x, hello.y, hello.z])
        dist = np.linalg.norm(node_pos - hello_pos)
        if dist <= 250 and hello.node_id != self.node_id:
            self.gnn.neighbors.add(hello.node_id)

    def construct_oppdata_frame(self, dst_id, hops=0, is_last_packet=False):
        # Build an opportunistic data frame to send to dst_id
        oppdata = OppData(self.node_id, dst_id, is_last_packet=is_last_packet)
        # For simplicity, reuse headers as in construct_frame
        multicast_addr = "224.1.1.1"
        broadcast_mac = "ff:ff:ff:ff:ff:ff"
        src_port = random.randint(1025, 65535)
        dst_port = 1024  # Different port for oppdata
        udp_payload = str(oppdata).encode()
        udp_length = 8 + len(udp_payload)
        udp_header = UDPHeader(src_port, dst_port, length=udp_length)
        ipv4_total_length = 20 + udp_length
        ipv4_header = IPv4Header(self.address, multicast_addr)
        ipv4_header.total_length = ipv4_total_length
        datalink = DataLink80211(self.mac, broadcast_mac)
        return {
            "datalink": datalink,
            "ipv4": ipv4_header,
            "udp": udp_header,
            "payload": oppdata,
            "hops": hops  # Track hop count
        }

    def receive_oppdata_frame(self, frame):
        oppdata = frame["payload"]
        # If this node is the destination, "receive" the data
        if oppdata.dst_id == self.node_id:
            # For demo, just print receipt
            print(f"Node {self.node_id} received OppData from {oppdata.src_id}: {oppdata.data:.3f} (hops={frame.get('hops', 0)}) {'(LAST)' if oppdata.is_last_packet else ''}")
            return True
        return False

    def forward_oppdata(self, config_nodes, spatial_grid, delivered_packets, max_hops=50):
        """Process the forwarding queue for multi-hop delivery."""
        new_queue = []
        for frame in self.forwarding_queue:
            oppdata = frame["payload"]
            hops = frame.get("hops", 0)
            if hops >= max_hops:
                new_queue.append(frame) # Re-add to queue if not delivered and hops exceeded
                continue
            if self.receive_oppdata_frame(frame):
                delivered_packets.append((oppdata.src_id, oppdata.dst_id, oppdata.is_last_packet)) # Modified to include is_last_packet
                continue  # Delivered
            # Not delivered, forward to next best neighbor
            self.gnn.update_neighbors()
            neighbor_scores = self.gnn.compute_embeddings()
            if neighbor_scores:
                # Only consider neighbors closer to destination
                dst_node = next((n for n in config_nodes if n.node_id == oppdata.dst_id), None)
                my_pos = np.array([self.x, self.y, self.z])
                dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
                progress_candidates = [
                    nid for nid in neighbor_scores
                    if np.linalg.norm(
                        np.array([
                            next((n for n in config.nodes if n.node_id == nid), None).x,
                            next((n for n in config.nodes if n.node_id == nid), None).y,
                            next((n for n in config.nodes if n.node_id == nid), None).z
                        ]) - dst_pos
                    ) < np.linalg.norm(my_pos - dst_pos)
                ]
                if progress_candidates:
                    best_forwarder_id = self.gnn.mab_select_forwarder(progress_candidates, dst_node=dst_node)
                    if best_forwarder_id in self.gnn.neighbors:
                        forwarder = next((n for n in config.nodes if n.node_id == best_forwarder_id), None)
                        if forwarder:
                            new_frame = self.construct_oppdata_frame(oppdata.dst_id, hops=hops+1, is_last_packet=oppdata.is_last_packet) # Pass is_last_packet
                            new_frame["payload"] = oppdata  # Preserve original data
                            forwarder.forwarding_queue.append(new_frame)
                            reward = 1.0 if forwarder.node_id == oppdata.dst_id else 0.0
                            self.gnn.mab_update(
                                best_forwarder_id, reward,
                                src_node=self,
                                dst_node=dst_node,
                                forwarder_node=forwarder
                            )
                            # Commented out for performance
                            # print(f"Node {self.node_id} forwarded OppData to {oppdata.dst_id} via {best_forwarder_id} (hops={hops+1})")
                        else:
                            new_queue.append(frame) # Re-add to queue if no forwarder found
                    else:
                        new_queue.append(frame) # Re-add to queue if best_forwarder_id is not a neighbor
                else:
                    new_queue.append(frame) # Re-add to queue if no progress candidates
            else:
                new_queue.append(frame) # Re-add to queue if no neighbor_scores
        self.forwarding_queue = new_queue  # Clear queue after processing

    def tgnn_predict_link_lifetime(self, neighbor_id):
        # Only as state encoder, not for routing
        tgnn_hist = self.tgnn_history.get(neighbor_id, None)
        if tgnn_hist and len(tgnn_hist) == 5:
            edge_index_seq = [torch.tensor([[0,1],[1,0]], dtype=torch.long).to(device) for _ in range(5)]
            x_seq = torch.stack([torch.tensor(f, dtype=torch.float32).unsqueeze(0) for f in tgnn_hist])  # (window, 1, feat)
            with torch.no_grad():
                pred = self.tgnn(x_seq, edge_index_seq)
                return float(pred[0])
        return None

    def tgnn_train_step(self):
        # Train on collected targets (batch size 1 for simplicity)
        if not self.tgnn_targets:
            return
        self.tgnn.train()
        losses = []
        for x_seq, edge_index_seq, target in self.tgnn_targets:
            pred = self.tgnn(x_seq, edge_index_seq)
            loss = F.mse_loss(pred, target)
            self.tgnn_optimizer.zero_grad()
            loss.backward()
            self.tgnn_optimizer.step()
            losses.append(loss.item())
        self.tgnn_targets.clear()
        self.tgnn.eval()
        return np.mean(losses) if losses else 0.0

class Configuration:
    def __init__(self, num_nodes=2):
        self.num_nodes = num_nodes
        self.nodes = []
        self.create_nodes()
        # Assign GNN to each node after all nodes are created
        for node in self.nodes:
            node.gnn = OnlineGNN(node, self.nodes)

    def create_nodes(self):
        for i in range(self.num_nodes):
            node = Node(
                node_id=i,
                address=f"192.168.1.{i+1}",
                mac_address=f"00:0a:95:9d:68:{i:02x}"
            )
            node.x = random.uniform(0, 1000)
            node.y = random.uniform(0, 1000)
            node.z = random.uniform(0, 1000)
            self.nodes.append(node)

class SteadyStateRandomWaypointMobility:
    def __init__(self, nodes, speed, bound=1000):
        self.nodes = nodes
        self.speed = speed
        self.bound = bound
        self.destinations = [self._random_point() for _ in nodes]

    def _random_point(self):
        return (
            random.uniform(0, self.bound),
            random.uniform(0, self.bound),
            random.uniform(0, self.bound)
        )

    def step(self, dt=1.0):
        for idx, node in enumerate(self.nodes):
            dest = self.destinations[idx]
            dx = dest[0] - node.x
            dy = dest[1] - node.y
            dz = dest[2] - node.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-6:
                # Arrived at destination, pick a new one
                self.destinations[idx] = self._random_point()
                continue
            move_dist = self.speed * dt
            if move_dist >= dist:
                # Arrive at destination this step
                node.x, node.y, node.z = dest
                self.destinations[idx] = self._random_point()
            else:
                # Move towards destination
                node.x += (dx / dist) * move_dist
                node.y += (dy / dist) * move_dist
                node.z += (dz / dist) * move_dist

class Hello:
    """Holds spatial information for a node."""
    def __init__(self, node_id, x, y, z):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Hello(node_id={self.node_id}, x={self.x}, y={self.y}, z={self.z})"

class OppData:
    """Holds opportunistic data for a node to send to a destination."""
    def __init__(self, src_id, dst_id, data=None, is_last_packet=False):
        self.src_id = src_id
        self.dst_id = dst_id
        self.data = data if data is not None else random.random()
        self.is_last_packet = is_last_packet # Added new attribute

    def __repr__(self):
        return f"OppData(src_id={self.src_id}, dst_id={self.dst_id}, data={self.data:.3f}, is_last_packet={self.is_last_packet})"

class IPv4Header:
    """Represents an IPv4 header."""
    def __init__(self, src_addr, dst_addr, identification=0, ttl=64, protocol=17):
        self.version = 4
        self.ihl = 5
        self.tos = 0
        self.total_length = 0  # To be set after payload is known
        self.identification = identification
        self.flags = 0
        self.fragment_offset = 0
        self.ttl = ttl
        self.protocol = protocol  # 17 for UDP
        self.header_checksum = 0  # To be calculated
        self.src_addr = src_addr
        self.dst_addr = dst_addr

    def __repr__(self):
        return (f"IPv4Header(src={self.src_addr}, dst={self.dst_addr}, id={self.identification}, "
                f"ttl={self.ttl}, protocol={self.protocol})")

class UDPHeader:
    """Represents a UDP header."""
    def __init__(self, src_port, dst_port, length=0):
        self.src_port = src_port
        self.dst_port = dst_port
        self.length = length  # To be set after payload is known
        self.checksum = 0     # To be calculated

    def __repr__(self):
        return f"UDPHeader(src_port={self.src_port}, dst_port={self.dst_port}, length={self.length})"

class DataLink80211:
    """Represents IEEE 802.11 data link layer header."""
    def __init__(self, src_mac, dst_mac, frame_control=0x0800):
        self.frame_control = frame_control
        self.duration = 0
        self.addr1 = dst_mac
        self.addr2 = src_mac
        self.addr3 = "ff:ff:ff:ff:ff:ff"  # BSSID or broadcast
        self.seq_ctrl = 0

    def __repr__(self):
        return f"DataLink80211(src_mac={self.addr2}, dst_mac={self.addr1})"

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def assign_nodes(self, nodes):
        self.grid.clear()
        for node in nodes:
            cell_x = int(node.x // self.cell_size)
            cell_y = int(node.y // self.cell_size)
            cell_z = int(node.z // self.cell_size)
            cell_index = (cell_x, cell_y, cell_z)
            self.grid.setdefault(cell_index, []).append(node)

    def get_receivers(self, sender, radius):
        cell_x = int(sender.x // self.cell_size)
        cell_y = int(sender.y // self.cell_size)
        cell_z = int(sender.z // self.cell_size)
        receivers = []
        # Check this cell and adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = (cell_x + dx, cell_y + dy, cell_z + dz)
                    for node in self.grid.get(neighbor_cell, []):
                        if node.node_id != sender.node_id:
                            dist = math.sqrt(
                                (sender.x - node.x) ** 2 +
                                (sender.y - node.y) ** 2 +
                                (sender.z - node.z) ** 2
                            )
                            if dist <= radius:
                                receivers.append(node)
        return receivers

def plot_node_positions(nodes, step, cmap_name='viridis'):
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    zs = [node.z for node in nodes]
    node_ids = [node.node_id for node in nodes]
    cmap = get_cmap(cmap_name)
    norm = plt.Normalize(min(node_ids), max(node_ids))
    colors = [cmap(norm(nid)) for nid in node_ids]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=colors, s=60)
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, str(node_ids[i]), fontsize=8, color='black')
    ax.set_title(f"Node Positions at Step {step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

class CTDGTracker:
    """
    Tracks the appearance and disappearance of edges (neighbor relationships) over continuous time.
    Stores for each node: {neighbor_id: [(start_time, end_time), ...]}
    """
    def __init__(self, nodes):
        self.nodes = nodes
        self.edge_intervals = {node.node_id: {} for node in nodes}  # node_id: {neighbor_id: [(start, end), ...]}
        self.active_edges = {node.node_id: {} for node in nodes}    # node_id: {neighbor_id: start_time}

    def update(self, current_time):
        # For each node, update edge intervals based on current neighbors
        for node in self.nodes:
            node_id = node.node_id
            current_neighbors = set(node.gnn.get_neighbors())
            prev_neighbors = set(self.active_edges[node_id].keys())

            # New edges (appeared)
            for neighbor_id in current_neighbors - prev_neighbors:
                self.active_edges[node_id][neighbor_id] = current_time

            # Edges that disappeared
            for neighbor_id in prev_neighbors - current_neighbors:
                start_time = self.active_edges[node_id].pop(neighbor_id)
                if neighbor_id not in self.edge_intervals[node_id]:
                    self.edge_intervals[node_id][neighbor_id] = []
                self.edge_intervals[node_id][neighbor_id].append((start_time, current_time))

    def finalize(self, final_time):
        # Close any open intervals at the end of simulation
        for node_id, neighbors in self.active_edges.items():
            for neighbor_id, start_time in neighbors.items():
                if neighbor_id not in self.edge_intervals[node_id]:
                    self.edge_intervals[node_id][neighbor_id] = []
                self.edge_intervals[node_id][neighbor_id].append((start_time, final_time))
        self.active_edges = {node_id: {} for node_id in self.active_edges}

    def get_edge_intervals(self, node_id, neighbor_id):
        return self.edge_intervals.get(node_id, {}).get(neighbor_id, [])

    def get_all_events(self):
        # Returns a list of (node_id, neighbor_id, start, end) for all edges
        events = []
        for node_id, neighbors in self.edge_intervals.items():
            for neighbor_id, intervals in neighbors.items():
                for start, end in intervals:
                    events.append((node_id, neighbor_id, start, end))
        return events

def inject_noise(delivered_packets, noise_level):
    """Randomly drop a fraction of delivered packets to simulate noise."""
    if noise_level <= 0.0:
        return delivered_packets
    keep_count = int((1.0 - noise_level) * len(delivered_packets))
    if keep_count <= 0:
        return []
    return random.sample(delivered_packets, keep_count)

if __name__ =="__main__":
    noise_levels = [0.1, 0.15, 0.25]
    noise_results = {}

    for noise_level in noise_levels:
        print(f"\n=== Running simulation with noise level: {noise_level} ===")
        config = Configuration(num_nodes=120)
        mobility = SteadyStateRandomWaypointMobility(config.nodes, speed=30)  # Increased speed
        cell_size = 350  # Match neighbor radius
        steps = 12  # Slightly more steps
        packets_per_step = 120  # More packets per step
        packet_size_bits = 8000 # 1KB packet size in bits
        time_per_step_seconds = 0.1 # 0.1 seconds per simulation step

        spatial_grid = SpatialGrid(cell_size)
        ctdg_tracker = CTDGTracker(config.nodes)

        pdr_list = []
        throughput_list = []
        gnn_loss_list = []
        mab_avg_reward_list = []
        reward_list = []
        cumulative_reward_list = []
        regret_list = []
        energy_consumption_list = []
        routing_overhead_list = []
        forwarding_attempts = 0  # Track actual forwarding attempts

        cumulative_reward = 0.0
        optimal_reward_per_step = packets_per_step  # Assume optimal = all packets delivered (reward=1 per packet)
        cumulative_optimal_reward = 0.0

        # # --- Pre-training phase for GNN ---
        # pretrain_steps = 2000  # More pre-training
        # for _ in range(pretrain_steps):
        #     mobility.step()
        #     spatial_grid.assign_nodes(config.nodes)
        #     for node in config.nodes:
        #         node.gnn.update_neighbors()
        #         feedback = {}
        #         valid_neighbors = set(node.gnn.neighbor_features.keys())
        #         for neighbor_id in valid_neighbors:
        #             dst = random.choice([n for n in config.nodes if n.node_id != node.node_id])
        #             my_pos = np.array([node.x, node.y, node.z])
        #             dst_pos = np.array([dst.x, dst.y, dst.z])
        #             neighbor = next((n for n in config.nodes if n.node_id == neighbor_id), None)
        #             neighbor_pos = np.array([neighbor.x, neighbor.y, neighbor.z])
        #             feedback[neighbor_id] = 1.0 if np.linalg.norm(neighbor_pos - dst_pos) < np.linalg.norm(my_pos - dst_pos) else 0.0
        #         node.gnn.online_update(feedback)
        #     # --- CTDG update for pretrain (optional, can skip if only main sim is tracked) ---
        #     for node in config.nodes:
        #         node.gnn.update_neighbors()
        #     ctdg_tracker.update(_ * time_per_step_seconds)  # Use pretrain time

        # --- Main simulation ---
        for step in range(steps):
            current_time = step * time_per_step_seconds # Assign a continuous time value
            mobility.step()
            spatial_grid.assign_nodes(config.nodes)

            for node in config.nodes:
                node.update_position_history(current_time) # Update node's own position history

            for sender in config.nodes:
                frame = sender.construct_frame()
                receivers = spatial_grid.get_receivers(sender, sender.gnn.neighbor_radius)
                current_neighbors_ids = {r.node_id for r in receivers} # Get current neighbors for link tracking
                sender.update_link_state_tracker(current_time, current_neighbors_ids, config.nodes)

            # --- Online TGNN training step for each node ---
            tgnn_losses = []
            for node in config.nodes:
                loss = node.tgnn_train_step()
                if loss is not None:
                    tgnn_losses.append(loss)
            # Optionally, you can log tgnn_losses if desired

            # --- CTDG update ---
            for node in config.nodes:
                node.gnn.update_neighbors()
            ctdg_tracker.update(current_time)

            # --- GNN performance metric: average MSE loss per node per step ---
            gnn_losses = []
            for node in config.nodes:
                node.gnn.update_neighbors()
                feedback = {}
                valid_neighbors = set(node.gnn.neighbor_features.keys())
                for neighbor_id in valid_neighbors:
                    dst = random.choice([n for n in config.nodes if n.node_id != node.node_id])
                    my_pos = np.array([node.x, node.y, node.z])
                    dst_pos = np.array([dst.x, dst.y, dst.z])
                    neighbor = next((n for n in config.nodes if n.node_id == neighbor_id), None)
                    neighbor_pos = np.array([neighbor.x, neighbor.y, neighbor.z])
                    feedback[neighbor_id] = 1.0 if np.linalg.norm(neighbor_pos - dst_pos) < np.linalg.norm(my_pos - dst_pos) else 0.0

                # --- Compute and store GNN loss ---
                if node.gnn.neighbor_features:
                    x, edge_index, id_to_idx = node.gnn._build_graph()
                    targets = torch.zeros(x.size(0)).to(device) # Move tensor to device
                    for nid, val in feedback.items():
                        idx = id_to_idx.get(nid, None)
                        if idx is not None and idx < len(targets):
                            targets[idx] = val
                    with torch.no_grad():
                        pred = node.gnn.gnn(x, edge_index)
                        loss = F.mse_loss(pred, targets)
                        gnn_losses.append(loss.item())
                node.gnn.online_update(feedback)
            gnn_loss_list.append(np.mean(gnn_losses) if gnn_losses else 0.0)

            # --- Data packet delivery ---
            delivered_packets = []
            packet_delays = []
            packet_sent_time = {}
            packet_hop_count = {}  # Track hops for each packet
            for _ in range(packets_per_step):
                src, dst = random.sample(config.nodes, 2)
                is_last = random.random() < 0.1
                opp_frame = src.construct_oppdata_frame(dst.node_id, hops=0, is_last_packet=is_last)
                src.forwarding_queue.append(opp_frame)
                packet_key = (src.node_id, dst.node_id, is_last)
                packet_sent_time[packet_key] = current_time
                packet_hop_count[packet_key] = 0

            # --- Forwarding with energy consumption and forwarding attempt tracking ---
            for node in config.nodes:
                new_queue = []
                for frame in node.forwarding_queue:
                    oppdata = frame["payload"]
                    hops = frame.get("hops", 0)
                    packet_key = (oppdata.src_id, oppdata.dst_id, oppdata.is_last_packet)
                    if packet_key not in packet_sent_time:
                        packet_sent_time[packet_key] = current_time
                    if packet_key not in packet_hop_count:
                        packet_hop_count[packet_key] = hops
                    if hops >= 50:
                        continue  # Drop packet if max hops exceeded
                    if node.receive_oppdata_frame(frame):
                        delivered_packets.append((oppdata.src_id, oppdata.dst_id, oppdata.is_last_packet))
                        continue
                    node.gnn.update_neighbors()
                    neighbor_scores = node.gnn.compute_embeddings()
                    if neighbor_scores:
                        dst_node = next((n for n in config.nodes if n.node_id == oppdata.dst_id), None)
                        my_pos = np.array([node.x, node.y, node.z])
                        dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
                        progress_candidates = [
                            nid for nid in neighbor_scores
                            if np.linalg.norm(
                                np.array([
                                    next((n for n in config.nodes if n.node_id == nid), None).x,
                                    next((n for n in config.nodes if n.node_id == nid), None).y,
                                    next((n for n in config.nodes if n.node_id == nid), None).z
                                ]) - dst_pos
                            ) < np.linalg.norm(my_pos - dst_pos)
                        ]
                        if progress_candidates:
                            best_forwarder_id = node.gnn.mab_select_forwarder(progress_candidates, dst_node=dst_node)
                            if best_forwarder_id in node.gnn.neighbors:
                                forwarder = next((n for n in config.nodes if n.node_id == best_forwarder_id), None)
                                if forwarder:
                                    new_frame = node.construct_oppdata_frame(oppdata.dst_id, hops=hops+1, is_last_packet=oppdata.is_last_packet)
                                    new_frame["payload"] = oppdata  # Preserve original data
                                    forwarder.forwarding_queue.append(new_frame)
                                    node.energy -= 0.01  # Transmission cost (tune as needed)
                                    forwarder.energy -= 0.005  # Reception cost (tune as needed)
                                    forwarding_attempts += 1
                                    packet_hop_count[packet_key] += 1
                                    reward = 1.0 if forwarder.node_id == oppdata.dst_id else 0.0
                                    node.gnn.mab_update(
                                        best_forwarder_id, reward,
                                        src_node=node,
                                        dst_node=dst_node,
                                        forwarder_node=forwarder
                                    )
                                else:
                                    new_queue.append(frame)
                            else:
                                new_queue.append(frame)
                        else:
                            new_queue.append(frame)
                    else:
                        new_queue.append(frame)
                node.forwarding_queue = new_queue

            # --- Inject noise here ---
            delivered_packets_noisy = inject_noise(delivered_packets, noise_level)

            # --- Throughput and PDR ---
            throughput_mbps = (len(delivered_packets_noisy) * packet_size_bits) / (time_per_step_seconds * 1e6)
            throughput_list.append(throughput_mbps)
            pdr = len(delivered_packets_noisy) / packets_per_step if packets_per_step > 0 else 0.0
            pdr_list.append(pdr)

            print(f"Step {step} (noise={noise_level}): PDR = {len(delivered_packets_noisy)}/{packets_per_step} ({pdr:.2f}), Throughput = {throughput_mbps:.3f} Mbps")

        avg_pdr = np.mean(pdr_list)
        avg_throughput = np.mean(throughput_list)
        noise_results[noise_level] = {
            "pdr_list": pdr_list,
            "throughput_list": throughput_list,
            "avg_pdr": avg_pdr,
            "avg_throughput": avg_throughput
        }
        print(f"=== Noise {noise_level}: Avg PDR = {avg_pdr:.3f}, Avg Throughput = {avg_throughput:.3f} Mbps ===")

    # --- Print summary table ---
    print("\n=== Summary of PDR and Throughput for Different Noise Levels ===")
    for noise_level in noise_levels:
        print(f"Noise {noise_level}: Avg PDR = {noise_results[noise_level]['avg_pdr']:.3f}, "
              f"Avg Throughput = {noise_results[noise_level]['avg_throughput']:.3f} Mbps")

    # --- Optionally, plot PDR and Throughput for each noise level ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    steps_range = range(steps)

    plt.figure(figsize=(7,4))
    for noise_level in noise_levels:
        plt.plot(steps_range, noise_results[noise_level]["pdr_list"], marker='o', label=f"Noise {noise_level}")
    plt.title("Packet Delivery Ratio (PDR) per Step (with Noise)")
    plt.xlabel("Step")
    plt.ylabel("PDR")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    for noise_level in noise_levels:
        plt.plot(steps_range, noise_results[noise_level]["throughput_list"], marker='o', label=f"Noise {noise_level}")
    plt.title("Throughput per Step (with Noise)")
    plt.xlabel("Step")
    plt.ylabel("Throughput (Mbps)")
    plt.legend()
    plt.tight_layout()
    plt.show()