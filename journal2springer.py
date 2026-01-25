# If you see "ModuleNotFoundError: No module named 'torch'", install torch first:
# pip install torch
# To use tgnn, install it with:
# pip install tgnn

# === Imports ===
import random
import math
import time
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import sys # Import sys for object size analysis
# import tgnn # Import TGNN for graph neural network-based forwarding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import NNConv, GCNConv
import csv
import os
import zlib  # <-- Add this import
from datetime import datetime  # <-- Add this import

# === Global constants ===
GNN_WEIGHTS_PATH = "gnn_forward_agent_weights.pt"

# --- Add global variable for noise level ---
NOISE_LEVEL = 0.0  # Default, will be set in simulation loop

# --- Add global simulation time variable ---
current_sim_time = 0.0  # Simulation time in seconds
SIM_TIME_STEP = 1.0     # Simulation time step per loop iteration (seconds)

# === Custom Online TGNN Model (NNConv + edge features) ===
class CustomOnlineTGNN(nn.Module):
    def __init__(self, node_feat_dim=12, edge_feat_dim=3, hidden_dim=64, lr=2e-4, dropout_p=0.05, train_every=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use much smaller hidden_dim to avoid OOM on CPU
        # Edge MLPs for each NNConv layer with correct output shape and minimal size
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim * hidden_dim, node_feat_dim * hidden_dim)
        ).to(self.device)
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        ).to(self.device)
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        ).to(self.device)
        self.edge_mlp4 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        ).to(self.device)
        self.conv1 = NNConv(node_feat_dim, hidden_dim, self.edge_mlp1, aggr='mean').to(self.device)
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr='mean').to(self.device)
        self.conv3 = NNConv(hidden_dim, hidden_dim, self.edge_mlp3, aggr='mean').to(self.device)
        self.conv4 = NNConv(hidden_dim, hidden_dim, self.edge_mlp4, aggr='mean').to(self.device)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, 1).to(self.device)
        self.ttl_head = nn.Linear(hidden_dim, 1).to(self.device)  # For adaptive TTL
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss(beta=0.3)
        self.experiences = []
        self.train_every = train_every
        self.early_stop_loss = 0.002

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        h = torch.relu(self.conv2(h, edge_index, edge_attr))
        h = torch.relu(self.conv3(h, edge_index, edge_attr))
        h = torch.relu(self.conv4(h, edge_index, edge_attr))
        h = self.dropout(h)
        return h

    def predict(self, node_feats, edge_index, edge_attr, candidate_indices):
        with torch.no_grad():
            x = torch.tensor(node_feats, dtype=torch.float, device=self.device)
            edge_idx = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            if edge_idx.dim() == 2 and edge_idx.shape[0] != 2:
                edge_idx = edge_idx.t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
            h = self.forward(x, edge_idx, edge_attr)
            scores = self.fc(h).squeeze()
            candidate_scores = scores[candidate_indices]
            return candidate_scores.detach().cpu().numpy()

    def predict_ttl(self, node_feats, edge_index, edge_attr):
        with torch.no_grad():
            x = torch.tensor(node_feats, dtype=torch.float, device=self.device)
            edge_idx = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            if edge_idx.dim() == 2 and edge_idx.shape[0] != 2:
                edge_idx = edge_idx.t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
            h = self.forward(x, edge_idx, edge_attr)
            ttl_pred = self.ttl_head(h[0]).squeeze()  # Use source node's embedding
            return max(10, float(ttl_pred.detach().cpu().item()))  # Clamp to min 10

    def store_experience(self, state, action, reward, next_state, done):
        self.experiences.append((state, action, reward, next_state, done))
        if len(self.experiences) >= self.train_every:
            self.train_on_experiences()
            self.experiences = []

    def train_on_experiences(self):
        if not self.experiences:
            return
        batch = self.experiences
        losses = []
        for state, action, reward, next_state, done in batch:
            node_feats, edge_index, edge_attr, candidate_indices = state
            x = torch.tensor(node_feats, dtype=torch.float, device=self.device)
            edge_idx = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            if edge_idx.dim() == 2 and edge_idx.shape[0] != 2:
                edge_idx = edge_idx.t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
            h = self.forward(x, edge_idx, edge_attr)
            scores = self.fc(h).squeeze()
            pred = scores[action]
            target = torch.tensor(reward, dtype=torch.float, device=self.device)
            loss = self.loss_fn(pred, target)
            losses.append(loss)
        if losses:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if total_loss.item() < self.early_stop_loss:
                return

    def predict_next_hop(self, current_node_id, nodes_features, edge_index, edge_attr, candidate_indices):
        with torch.no_grad():
            x = torch.tensor(nodes_features, dtype=torch.float, device=self.device)
            edge_idx = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            if edge_idx.dim() == 2 and edge_idx.shape[0] != 2:
                edge_idx = edge_idx.t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
            h = self.forward(x, edge_idx, edge_attr)
            scores = self.fc(h).squeeze()
            candidate_scores = scores[candidate_indices]
            if len(candidate_scores) == 0:
                return None
            best_idx = candidate_indices[int(np.argmax(candidate_scores))]
            return nodes_features[best_idx][0]

# === Packet Classes ===
class OppPacket:
    def __init__(self, source_ip, destination_ip, ttl, source_mac_address, source_x, source_y, source_z):
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.ttl = ttl
        self.creation_timestamp = None  # Will be set at creation using current_sim_time
        self.delivery_timestamp = None  # Will be set at delivery using current_sim_time
        self.initial_ttl = ttl
        self.delivered = False
        self.current_hop_mac = source_mac_address
        self.source_x = source_x
        self.source_y = source_y
        self.source_z = source_z

class HelloPacket:
    def __init__(self, source_ip, source_mac, x, y, z, timestamp, distance, node):
        self.source_ip = source_ip
        self.source_mac = source_mac
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = timestamp
        self.distance = distance
        self.node = node

    def __repr__(self):
        return f"HelloPacket(src_ip={self.source_ip}, distance={self.distance:.2f}, time={self.timestamp}, node_id={self.node.node_id})"

class DistanceUpdatePacket:
    def __init__(self, source_ip, destination_ip, distance, visited_nodes_bloom_filter, original_opp_destination_ip):
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.distance = distance
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.visited_nodes = visited_nodes_bloom_filter
        self.original_opp_destination_ip = original_opp_destination_ip
        self.ttl = 50

# === Node Class ===
class Node:
    packet_sent_count = 0
    packet_delivered_count = 0
    drop_count = 0
    delivered_packets = []
    delivered_packet_sizes = []
    delivered_visited_nodes_sizes = [] # Will be unused, can be removed

    def __init__(self, ip_address, mac_address, x, y, z, node_id):
        self.mac_address = mac_address
        self.node_id = self.generate_unique_node_id()
        self.ip_address = ip_address
        self.x = x
        self.y = y
        self.z = z
        self.queue = defaultdict(lambda: deque(maxlen=5))
        self.opp_packet_queue = deque(maxlen=500)  # Increased queue size for opportunistic packets
        self.distance_update_queue = deque()
        self.position = np.array([{
            'x': x, 'y': y, 'z': z,
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        self.known_destinations = {}
        self.energy = 100.0
        self.reward_by_direction = defaultdict(list)
        self.success_by_direction = defaultdict(lambda: {"success": 0, "total": 0})
        self.opp_dest_packet = []
        self.packets = np.array([])
        if not hasattr(Node, 'tgnn_model'):
            Node.tgnn_model = CustomOnlineTGNN()
        self.last_tgnn_state = None
        self.last_tgnn_action = None
        # Remove offline_forward_model attribute
        # if not hasattr(Node, 'offline_forward_model'):
        #     Node.offline_forward_model = None
        self.gnn_agent = None  # Will be set after all nodes are created

    def generate_unique_node_id(self):
        """Generates a unique 16-bit node ID using CRC32 of MAC and a random salt."""
        salt = random.getrandbits(32) # Generate a random 32-bit salt
        data = f"{self.mac_address}{salt}".encode('utf-8')
        crc32 = zlib.crc32(data) # Calculate CRC32
        return crc32 % 1000 # Ensure the ID is within the range [0, 999], suitable for Bloom filter capacity


    def hello(self, time, distance, node):
        """Creates a Hello packet."""
        pkt = {
            'source_ip': self.ip_address,
            'source_mac': self.mac_address,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'timestamp': time,
            'distance': distance,
            'node': self # Include the node object itself
        }
        # Create a dynamic object with attributes from the dictionary
        return type('HelloPacket', (object,), pkt)()

    def sort_mobility(self):
        """Sorts the position log by time."""
        self.position = np.array(sorted(
            self.position,
            key=lambda p: datetime.strptime(p['Time'], "%Y-%m-%d %H:%M:%S")
        ))

    def update_position_log(self):
        """Logs the current position and timestamp."""
        self.position = np.append(
            self.position,
            {
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    def distance_to(self, other_node):
        """Calculates the Euclidean distance to another node."""
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2 + (self.z - other_node.z)**2)

    def recent_distance(self, x, y, z):
        """Calculates the distance to a given coordinate."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)

    def node_direction(self, pkts):
        """Determines the direction of movement based on recent hello packets."""
        # Check if there are at least two packets before accessing indices
        if len(pkts) < 2:
            return 0  # Return a default direction (e.g., stationary) if not enough data
        return +1 if pkts[1].distance < pkts[0].distance else -1 # +1 if getting closer, -1 if moving away

    def relative_speed(self, pkts):
        """Calculates the relative speed to a neighbor."""
        # Check if there are at least two packets before accessing indices
        if len(pkts) < 2:
            return 0.0 # Return a default speed (0) if not enough data
        t1 = pd.to_datetime(pkts[0].timestamp)
        t2 = pd.to_datetime(pkts[1].timestamp)
        d1 = pkts[0].distance
        d2 = pkts[1].distance
        time_diff = (t2 - t1).total_seconds()
        return abs(d2 - d1) / time_diff if time_diff != 0 else 0

    def clean_stale_hello_packets(self, max_age_sec=20):
        """Removes stale hello packets from the queue."""
        now = pd.to_datetime(datetime.now())
        for neighbor_ip in list(self.queue.keys()):
            self.queue[neighbor_ip] = deque(
                [pkt for pkt in self.queue[neighbor_ip] if hasattr(pkt, 'timestamp') and (now - pd.to_datetime(pkt.timestamp)).total_seconds() <= max_age_sec],
                maxlen=5
            )

    def receive_hello_packet(self, hello_pkt):
        """Processes a received hello packet."""
        peer_ip = hello_pkt.source_ip
        self.queue[peer_ip].append(hello_pkt)
        # Update known distance if the hello packet is from a known destination
        if hello_pkt.source_ip in self.known_destinations:
            self.known_destinations[hello_pkt.source_ip] = hello_pkt.distance

    def send_distance_update(self, update_packet):
        """Sends a distance update packet (adds to queue for processing)."""
        # For simplicity, just add to the queue for now.
        # Actual routing back to source would be more complex in a real network.
        self.distance_update_queue.append(update_packet)

    def process_distance_updates(self):
        """Processes received distance update packets."""
        processed_updates = []
        while self.distance_update_queue:
            update_packet = self.distance_update_queue.popleft()

            # Decrement TTL of the distance update packet
            update_packet.ttl -= 1
            if update_packet.ttl <= 0:
                # Drop distance update packet if TTL expired
                continue

            # If this node is the original source of the packet that triggered this update
            if update_packet.destination_ip == self.ip_address:
                self.known_destinations[update_packet.original_opp_destination_ip] = update_packet.distance

            else:
                # --- Step 5: Intermediate node attempts to forward the distance update packet ---
                # This is an intermediate node receiving an update packet meant for the original source.
                # This node needs to attempt to forward the update packet towards update_packet.destination_ip
                # (the original source).

                # Find potential forwarders for the distance update packet (towards the original source)
                # Prioritize neighbors closer to the original source
                original_source_node = next((n for n in config.nodes if n.ip_address == update_packet.destination_ip), None)
                if original_source_node:
                    forwarder_candidates = [
                        pkt.node for neighbor_ip, pkts in self.queue.items() for pkt in pkts
                        if self.distance_to(pkt.node) <= 250
                        and self.energy > 1
                        and pkt.node.distance_to(original_source_node) < self.distance_to(original_source_node)
                    ]

                    if forwarder_candidates:
                        # Select the best forwarder (e.g., the one closest to the original source)
                        next_hop_for_update = min(
                            forwarder_candidates,
                            key=lambda node: node.distance_to(original_source_node)
                        )
                        # Attempt to forward the update packet to the next hop
                        # In a real simulation, this would involve adding to the next hop's queue
                        next_hop_for_update.distance_update_queue.append(update_packet)
                    else:
                        # No suitable forwarder found, the update packet might be dropped or re-queued
                        # For now, let's re-queue it, hoping for better opportunities later
                         processed_updates.append(update_packet) # Re-queue if not forwarded
                         # print(f"Node {self.node_id} could not forward DistanceUpdate towards {original_source_node.node_id}, re-queuing.")
                else:
                    # Original source node not found in config.nodes (shouldn't happen in this simulation)
                    # Drop the update packet
                    # print(f"Node {self.node_id} received DistanceUpdate for unknown source {update_packet.destination_ip}, dropping.")
                    pass
                # ------------------------------------------------------------------------------------------

        self.distance_update_queue.extend(processed_updates) # Add updates back that weren't processed/dropped


    def send_packet(self, destination_ip):
        """Creates and queues an opportunistic packet for sending with adaptive TTL."""
        destination_node = next((n for n in config.nodes if n.ip_address == destination_ip), None)
        if destination_node:
            # Only send if distance to destination is greater than 250 meters
            base_distance = self.distance_to(destination_node)
            if base_distance <= 250:
                return False
            # --- Adaptive TTL based on neighbor count and TGNN suggestion ---
            neighbor_count = sum(1 for pkts in self.queue.values() for _ in pkts)
            # Prepare dummy features for TTL prediction
            nodes_features = [
                [self.node_id, self.x, self.y, self.z, self.energy, 1.0, 0, 0, 0, self.energy, 0, base_distance]
            ]
            edge_index = []
            edge_attr = []
            idx = 1
            for neighbor_ip, pkts in self.queue.items():
                for pkt in pkts:
                    neighbor = pkt.node
                    nodes_features.append([
                        neighbor.node_id, neighbor.x, neighbor.y, neighbor.z, neighbor.energy, 0.0, 0, 0, 0, neighbor.energy, 0, neighbor.distance_to(destination_node)
                    ])
                    edge_index.append([0, idx])
                    # Use dummy edge features for TTL prediction
                    edge_attr.append([0.0, 0.0, 0.0])
                    idx += 1
            # Use TGNN to predict TTL if enough neighbors, else fallback
            if len(nodes_features) > 1:
                tgnn_ttl = Node.tgnn_model.predict_ttl(nodes_features, edge_index, edge_attr)
                ttl = int(min(300, max(30, tgnn_ttl)))
            else:
                # Fallback: density-based TTL
                ttl = int(min(300, max(30, base_distance / 5 + neighbor_count * 2)))
            packet = self.create_packet(destination_ip, ttl, self.mac_address, self.x, self.y, self.z)
            if packet:
                Node.packet_sent_count += 1
                # Set creation_timestamp using simulation time
                packet.creation_timestamp = current_sim_time
                self.opp_packet_queue.append(packet)
                return True
        return False

    def create_packet(self, destination_ip, ttl, source_mac_address, source_x, source_y, source_z): # Added source_mac_address and position
        """Creates an OppPacket instance."""
        packet = OppPacket(
            source_ip=self.ip_address,
            destination_ip=destination_ip,
            ttl=ttl,
            source_mac_address=source_mac_address, # Pass the source MAC
            source_x=source_x,
            source_y=source_y,
            source_z=source_z
        )
        # Set creation_timestamp using simulation time (handled in send_packet)
        # packet.creation_timestamp = datetime.now()  # REMOVE
        return packet

    def process_queue(self):
        """Processes packets in the opportunistic packet queue using TGNN for forwarding."""
        processed_packets = []
        for _ in range(len(self.opp_packet_queue)):
            packet = self.opp_packet_queue.popleft()
            if packet.delivered:
                continue
            if packet.ttl <= 0:
                if not packet.delivered:
                    Node.drop_count += 1
                continue
            success = self.tgnn_forward(packet)
            if not success:
                packet.ttl -= 1
                if packet.ttl > 0:
                    processed_packets.append(packet)
                else:
                    if not packet.delivered:
                        Node.drop_count += 1
        self.opp_packet_queue.extend(processed_packets)

    def tgnn_forward(self, opp_packet):
        """Uses GNN-based neighbor scoring for next-hop selection (progress-only)."""
        current_node = self
        opp_packet.current_hop_mac = current_node.mac_address

        # --- Simulate channel noise: randomly drop packet with probability NOISE_LEVEL ---
        if NOISE_LEVEL > 0.0 and random.random() < NOISE_LEVEL:
            Node.drop_count += 1
            return False  # Packet dropped due to noise

        # 1. Check if destination
        if current_node.ip_address == opp_packet.destination_ip:
            if hasattr(opp_packet, "delivered") and opp_packet.delivered:
                return True
            opp_packet.delivered = True
            Node.packet_delivered_count += 1
            # Set delivery_timestamp using simulation time
            opp_packet.delivery_timestamp = current_sim_time
            print(f"Packet delivered! Source: {opp_packet.source_ip}, Dest: {opp_packet.destination_ip}, Created: {opp_packet.creation_timestamp}, Delivered at: {opp_packet.delivery_timestamp}")

            Node.delivered_packets.append({
                'initial_ttl': opp_packet.initial_ttl,
                'final_ttl': opp_packet.ttl,
                'hops_used': opp_packet.initial_ttl - opp_packet.ttl,
                'creation_timestamp': opp_packet.creation_timestamp,
                'delivery_timestamp': opp_packet.delivery_timestamp
            })

            packet_size = sys.getsizeof(opp_packet) + sum(sys.getsizeof(attr_value) for attr_value in opp_packet.__dict__.values())
            Node.delivered_packet_sizes.append(packet_size)

            current_node.opp_dest_packet.append(opp_packet)
            current_node.packets = np.append(current_node.packets, opp_packet)

            # --- Step 3: Create and send distance update back to source ---
            distance_back_to_source = math.sqrt(
                (current_node.x - opp_packet.source_x)**2 +
                (current_node.y - opp_packet.source_y)**2 +
                (current_node.z - opp_packet.source_z)**2
            )

            update_pkt = DistanceUpdatePacket(
                source_ip=current_node.ip_address,
                destination_ip=opp_packet.source_ip,
                distance=distance_back_to_source,
                visited_nodes_bloom_filter=None, # Pass None, not used
                original_opp_destination_ip=opp_packet.destination_ip
            )
            current_node.send_distance_update(update_pkt)

            return True # Packet delivered

        # 2. Check if TTL expired
        if opp_packet.ttl <= 0:
            Node.drop_count += 1
            return False # Packet dropped

        # 3. Gather neighbor info for GNN input
        self.gnn_agent.update_neighbors()
        dst_node = next((n for n in config.nodes if n.ip_address == opp_packet.destination_ip), None)
        if not self.gnn_agent.neighbors or dst_node is None:
            Node.drop_count += 1
            return False

        # Only consider neighbors that make progress toward the destination
        next_node = self.gnn_agent.select_best_forwarder(dst_node)
        if next_node and next_node.energy > 0 and self.energy > 0:
            self.energy = max(0, self.energy - 0.0003)
            next_node.energy = max(0, next_node.energy - 0.0001)
            next_node.opp_packet_queue.append(opp_packet)
            # Optionally: online GNN update (reward = 1 for delivery, 0 for not delivered)
            # Feedback can be shaped as in reference if desired
            return True
        else:
            Node.drop_count += 1
            return False

# === Custom GRU Cell Implementation ===
class CustomGRUCell(nn.Module):
    """
    Custom GRU cell for MANET routing with learnable trust, congestion, and direction factors.
    Features:
      - All input features normalized before concatenation.
      - Learnable scaling for trust, congestion, and direction factors.
      - Robust to missing/noisy features via masking/defaults.
      - Dropout and extra layers for regularization and expressiveness.
      - Proper batch support and variable-length handling.
      - Direction factor: +1 (closer), 0 (stationary), -1 (away).
    Args:
      emb_dim: int, node embedding dimension.
      mob_dim: int, mobility feature dimension.
      hidden_dim: int, hidden state dimension.
      dropout_p: float, dropout probability.
    """
    def __init__(self, emb_dim, mob_dim=2, hidden_dim=64, dropout_p=0.1):
        super().__init__()
        self.input_dim = emb_dim + mob_dim + 1 + 1 + 1 + 1 + 1 + 1
        self.hidden_dim = hidden_dim

        # Learnable scaling factors for each gate
        self.energy_weight = nn.Parameter(torch.tensor(1.0))
        self.time_weight = nn.Parameter(torch.tensor(1.0))
        self.mobility_weight = nn.Parameter(torch.tensor(1.0))
        self.congestion_weight = nn.Parameter(torch.tensor(1.0))
        self.trust_weight = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout_p)
        self.linear_z1 = nn.Linear(self.input_dim + hidden_dim, hidden_dim)
        self.linear_z2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_r1 = nn.Linear(self.input_dim + hidden_dim, hidden_dim)
        self.linear_r2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h1 = nn.Linear(self.input_dim + hidden_dim, hidden_dim)
        self.linear_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(self.input_dim + hidden_dim)

    def normalize(self, x, mask=None, default=0.0):
        # Normalize to [0,1] if possible, else fallback to default
        if mask is not None:
            x = torch.where(mask, x, torch.tensor(default, device=x.device, dtype=x.dtype))
        x_min = torch.amin(x, dim=0, keepdim=True)
        x_max = torch.amax(x, dim=0, keepdim=True)
        denom = (x_max - x_min).clamp(min=1e-6)
        return (x - x_min) / denom

    def forward(self, input_emb, mobility, energy, drop_prob, time_feat, trust_score, queue_length, direction, h_prev, mask=None):
        """
        Args:
          input_emb: [batch, emb_dim]
          mobility: [batch, mob_dim]
          energy: [batch, 1]
          drop_prob: [batch, 1]
          time_feat: [batch, 1]
          trust_score: [batch, 1]
          queue_length: [batch, 1]
          direction: [batch, 1] (+1, 0, -1)
          h_prev: [batch, hidden_dim]
          mask: [batch, input_dim] or None, True for valid, False for missing
        Returns:
          h_new: [batch, hidden_dim]
        """
        # Robust normalization and masking
        input_emb = self.normalize(input_emb)
        mobility = self.normalize(mobility)
        energy = self.normalize(energy)
        drop_prob = self.normalize(drop_prob)
        time_feat = self.normalize(time_feat)
        trust_score = self.normalize(trust_score)
        queue_length = self.normalize(queue_length)
        # Direction: map +1->1, 0->0.5, -1->0
        direction_norm = (direction + 1) / 2.0  # +1=1, 0=0.5, -1=0
        direction_norm = direction_norm.clamp(0, 1)

        # Handle missing features (mask or default)
        if mask is not None:
            input_emb = torch.where(mask[:, :input_emb.shape[1]], input_emb, torch.zeros_like(input_emb))
            mobility = torch.where(mask[:, input_emb.shape[1]:input_emb.shape[1]+mobility.shape[1]], mobility, torch.zeros_like(mobility))
            # ...repeat for other features as needed...

        # Concatenate all normalized features
        x = torch.cat([input_emb, mobility, energy, drop_prob, time_feat, trust_score, queue_length, direction_norm], dim=-1)
        x = self.dropout(x)
        combined = torch.cat([x, h_prev], dim=-1)
        combined = self.ln(combined)

        # Standard GRU gates
        z = torch.sigmoid(self.linear_z2(F.gelu(self.linear_z1(combined))))
        r = torch.sigmoid(self.linear_r2(F.gelu(self.linear_r1(combined))))
        combined_reset = torch.cat([x, r * h_prev], dim=-1)
        combined_reset = self.ln(combined_reset)
        h_tilde = torch.tanh(self.linear_h2(F.gelu(self.linear_h1(combined_reset))))
        h_tilde = self.dropout(h_tilde)

        # MANET-aware gates
        # Energy Gate
        energy_gate = torch.sigmoid(self.energy_weight * energy)
        # Time Gate (exponential decay, assume time_feat is Δt normalized)
        time_decay = torch.exp(-time_feat)
        time_gate = torch.sigmoid(self.time_weight * time_decay)
        # Mobility Gate (relative speed + direction)
        mobility_feat = torch.cat([mobility, direction_norm], dim=-1)
        mobility_gate = torch.sigmoid(self.mobility_weight * mobility_feat.mean(dim=-1, keepdim=True))
        # Congestion Gate (inverse queue occupancy)
        congestion_gate = torch.sigmoid(self.congestion_weight * (1 - queue_length))
        # Trust Gate
        trust_gate = torch.sigmoid(self.trust_weight * trust_score)

        # Unified gate modulation
        update_gate = z * energy_gate * time_gate * mobility_gate * congestion_gate * trust_gate
        reset_gate = r * energy_gate * mobility_gate * trust_gate
        candidate_gate = h_tilde * energy_gate * mobility_gate * trust_gate

        # Final hidden state update
        h_new = (1 - update_gate) * h_prev + update_gate * candidate_gate
        return h_new

# === GCN-based GNN for neighbor scoring (from reference) ===
class GCNNeighborScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.conv_out = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(0.18)

    def forward(self, x, edge_index):
        x1 = F.gelu(self.ln1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        x2 = F.gelu(self.ln2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        x3 = F.gelu(self.ln3(self.conv3(x2 + x1, edge_index)))  # Residual
        x3 = self.dropout(x3)
        out = self.conv_out(x3, edge_index)
        return out.squeeze(-1)

# === Per-node GNN agent for neighbor scoring ===
class GNNForwardAgent:
    def __init__(self, node, all_nodes, input_dim=5, hidden_dim=64, neighbor_radius=300):
        self.node = node
        self.all_nodes = all_nodes
        self.neighbor_radius = neighbor_radius
        self.gnn = GCNNeighborScorer(input_dim, hidden_dim).to(torch.device("cpu"))
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.002)
        self.last_x = None
        self.last_edge_index = None
        # Load weights if available
        self.load_weights()

    def load_weights(self):
        if os.path.exists(GNN_WEIGHTS_PATH):
            try:
                state = torch.load(GNN_WEIGHTS_PATH, map_location="cpu")
                self.gnn.load_state_dict(state)
                print(f"Loaded GNN weights from {GNN_WEIGHTS_PATH}")
            except Exception as e:
                print(f"Failed to load GNN weights: {e}")

    def save_weights(self):
        try:
            torch.save(self.gnn.state_dict(), GNN_WEIGHTS_PATH)
            print(f"Saved GNN weights to {GNN_WEIGHTS_PATH}")
        except Exception as e:
            print(f"Failed to save GNN weights: {e}")

    def update_neighbors(self):
        self.neighbors = []
        self.neighbor_features = {}
        node_pos = np.array([self.node.x, self.node.y, self.node.z])
        for other in self.all_nodes:
            if other.node_id == self.node.node_id:
                continue
            other_pos = np.array([other.x, other.y, other.z])
            dist = np.linalg.norm(node_pos - other_pos)
            if dist <= self.neighbor_radius:
                self.neighbors.append(other)
                feat = np.array([
                    other.x, other.y, other.z,
                    getattr(other, 'energy', 100.0),
                    dist
                ], dtype=np.float32)
                self.neighbor_features[other.node_id] = feat

    def _build_graph(self):
        node_ids = [self.node.node_id] + [n.node_id for n in self.neighbors]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        x = [np.array([self.node.x, self.node.y, self.node.z, getattr(self.node, 'energy', 100.0), 0.0], dtype=np.float32)]
        x += [self.neighbor_features[n.node_id] for n in self.neighbors]
        x = torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32)
        edge_index = []
        for n in self.neighbors:
            i = 0
            j = id_to_idx[n.node_id]
            edge_index.append([i, j])
            edge_index.append([j, i])
        if not edge_index:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return x, edge_index, id_to_idx

    def compute_scores(self):
        if not self.neighbors:
            return {}
        x, edge_index, id_to_idx = self._build_graph()
        self.last_x = x
        self.last_edge_index = edge_index
        with torch.no_grad():
            scores = self.gnn(x, edge_index)
        return {n.node_id: float(scores[id_to_idx[n.node_id]]) for n in self.neighbors}

    def select_best_forwarder(self, dst_node):
        # Only consider neighbors that are closer to the destination
        my_pos = np.array([self.node.x, self.node.y, self.node.z])
        dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
        scores = self.compute_scores()
        progress_candidates = []
        for n in self.neighbors:
            n_pos = np.array([n.x, n.y, n.z])
            if np.linalg.norm(n_pos - dst_pos) < np.linalg.norm(my_pos - dst_pos):
                progress_candidates.append((n.node_id, scores.get(n.node_id, -float('inf'))))
        if not progress_candidates:
            return None
        # Pick the neighbor with the highest GNN score among progress candidates
        best_nid = max(progress_candidates, key=lambda x: x[1])[0]
        return next((n for n in self.neighbors if n.node_id == best_nid), None)

    def online_update(self, feedback):
        if not self.neighbors or not feedback:
            return
        if self.last_x is None or self.last_edge_index is None:
            return
        node_ids = [self.node.node_id] + [n.node_id for n in self.neighbors]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        targets = torch.zeros(self.last_x.size(0))
        for nid, val in feedback.items():
            idx = id_to_idx.get(nid, None)
            if idx is not None and idx < len(targets):
                targets[idx] = val
        pred = self.gnn(self.last_x, self.last_edge_index)
        loss = F.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Save weights after online update
        self.save_weights()

# === Patch Node to use GNNForwardAgent for forwarding ===
class Node:
    packet_sent_count = 0
    packet_delivered_count = 0
    drop_count = 0
    delivered_packets = [] # List to store info about delivered packets
    # Add lists to store packet and visited_nodes sizes for delivered packets
    delivered_packet_sizes = []
    delivered_visited_nodes_sizes = [] # Will be unused, can be removed


    def __init__(self, ip_address, mac_address, x, y, z, node_id):
        self.mac_address = mac_address
        self.node_id = self.generate_unique_node_id()
        self.ip_address = ip_address
        self.x = x
        self.y = y
        self.z = z
        self.queue = defaultdict(lambda: deque(maxlen=5))
        self.opp_packet_queue = deque(maxlen=500)  # Increased queue size for opportunistic packets
        self.distance_update_queue = deque()
        self.position = np.array([{
            'x': x, 'y': y, 'z': z,
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        self.known_destinations = {}
        self.energy = 100.0
        self.reward_by_direction = defaultdict(list)
        self.success_by_direction = defaultdict(lambda: {"success": 0, "total": 0})
        self.opp_dest_packet = []
        self.packets = np.array([])
        if not hasattr(Node, 'tgnn_model'):
            Node.tgnn_model = CustomOnlineTGNN()
        self.last_tgnn_state = None
        self.last_tgnn_action = None
        # Remove offline_forward_model attribute
        # if not hasattr(Node, 'offline_forward_model'):
        #     Node.offline_forward_model = None
        self.gnn_agent = None  # Will be set after all nodes are created

    def generate_unique_node_id(self):
        """Generates a unique 16-bit node ID using CRC32 of MAC and a random salt."""
        salt = random.getrandbits(32) # Generate a random 32-bit salt
        data = f"{self.mac_address}{salt}".encode('utf-8')
        crc32 = zlib.crc32(data) # Calculate CRC32
        return crc32 % 1000 # Ensure the ID is within the range [0, 999], suitable for Bloom filter capacity


    def hello(self, time, distance, node):
        """Creates a Hello packet."""
        pkt = {
            'source_ip': self.ip_address,
            'source_mac': self.mac_address,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'timestamp': time,
            'distance': distance,
            'node': self # Include the node object itself
        }
        # Create a dynamic object with attributes from the dictionary
        return type('HelloPacket', (object,), pkt)()

    def sort_mobility(self):
        """Sorts the position log by time."""
        self.position = np.array(sorted(
            self.position,
            key=lambda p: datetime.strptime(p['Time'], "%Y-%m-%d %H:%M:%S")
        ))

    def update_position_log(self):
        """Logs the current position and timestamp."""
        self.position = np.append(
            self.position,
            {
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    def distance_to(self, other_node):
        """Calculates the Euclidean distance to another node."""
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2 + (self.z - other_node.z)**2)

    def recent_distance(self, x, y, z):
        """Calculates the distance to a given coordinate."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)

    def node_direction(self, pkts):
        """Determines the direction of movement based on recent hello packets."""
        # Check if there are at least two packets before accessing indices
        if len(pkts) < 2:
            return 0  # Return a default direction (e.g., stationary) if not enough data
        return +1 if pkts[1].distance < pkts[0].distance else -1 # +1 if getting closer, -1 if moving away

    def relative_speed(self, pkts):
        """Calculates the relative speed to a neighbor."""
        # Check if there are at least two packets before accessing indices
        if len(pkts) < 2:
            return 0.0 # Return a default speed (0) if not enough data
        t1 = pd.to_datetime(pkts[0].timestamp)
        t2 = pd.to_datetime(pkts[1].timestamp)
        d1 = pkts[0].distance
        d2 = pkts[1].distance
        time_diff = (t2 - t1).total_seconds()
        return abs(d2 - d1) / time_diff if time_diff != 0 else 0

    def clean_stale_hello_packets(self, max_age_sec=20):
        """Removes stale hello packets from the queue."""
        now = pd.to_datetime(datetime.now())
        for neighbor_ip in list(self.queue.keys()):
            self.queue[neighbor_ip] = deque(
                [pkt for pkt in self.queue[neighbor_ip] if hasattr(pkt, 'timestamp') and (now - pd.to_datetime(pkt.timestamp)).total_seconds() <= max_age_sec],
                maxlen=5
            )

    def receive_hello_packet(self, hello_pkt):
        """Processes a received hello packet."""
        peer_ip = hello_pkt.source_ip
        self.queue[peer_ip].append(hello_pkt)
        # Update known distance if the hello packet is from a known destination
        if hello_pkt.source_ip in self.known_destinations:
            self.known_destinations[hello_pkt.source_ip] = hello_pkt.distance

    def send_distance_update(self, update_packet):
        """Sends a distance update packet (adds to queue for processing)."""
        # For simplicity, just add to the queue for now.
        # Actual routing back to source would be more complex in a real network.
        self.distance_update_queue.append(update_packet)

    def process_distance_updates(self):
        """Processes received distance update packets."""
        processed_updates = []
        while self.distance_update_queue:
            update_packet = self.distance_update_queue.popleft()

            # Decrement TTL of the distance update packet
            update_packet.ttl -= 1
            if update_packet.ttl <= 0:
                # Drop distance update packet if TTL expired
                continue

            # If this node is the original source of the packet that triggered this update
            if update_packet.destination_ip == self.ip_address:
                self.known_destinations[update_packet.original_opp_destination_ip] = update_packet.distance

            else:
                # --- Step 5: Intermediate node attempts to forward the distance update packet ---
                # This is an intermediate node receiving an update packet meant for the original source.
                # This node needs to attempt to forward the update packet towards update_packet.destination_ip
                # (the original source).

                # Find potential forwarders for the distance update packet (towards the original source)
                # Prioritize neighbors closer to the original source
                original_source_node = next((n for n in config.nodes if n.ip_address == update_packet.destination_ip), None)
                if original_source_node:
                    forwarder_candidates = [
                        pkt.node for neighbor_ip, pkts in self.queue.items() for pkt in pkts
                        if self.distance_to(pkt.node) <= 250
                        and self.energy > 1
                        and pkt.node.distance_to(original_source_node) < self.distance_to(original_source_node)
                    ]

                    if forwarder_candidates:
                        # Select the best forwarder (e.g., the one closest to the original source)
                        next_hop_for_update = min(
                            forwarder_candidates,
                            key=lambda node: node.distance_to(original_source_node)
                        )
                        # Attempt to forward the update packet to the next hop
                        # In a real simulation, this would involve adding to the next hop's queue
                        next_hop_for_update.distance_update_queue.append(update_packet)
                    else:
                        # No suitable forwarder found, the update packet might be dropped or re-queued
                        # For now, let's re-queue it, hoping for better opportunities later
                         processed_updates.append(update_packet) # Re-queue if not forwarded
                         # print(f"Node {self.node_id} could not forward DistanceUpdate towards {original_source_node.node_id}, re-queuing.")
                else:
                    # Original source node not found in config.nodes (shouldn't happen in this simulation)
                    # Drop the update packet
                    # print(f"Node {self.node_id} received DistanceUpdate for unknown source {update_packet.destination_ip}, dropping.")
                    pass
                # ------------------------------------------------------------------------------------------

        self.distance_update_queue.extend(processed_updates) # Add updates back that weren't processed/dropped


    def send_packet(self, destination_ip):
        """Creates and queues an opportunistic packet for sending with adaptive TTL."""
        destination_node = next((n for n in config.nodes if n.ip_address == destination_ip), None)
        if destination_node:
            # Only send if distance to destination is greater than 250 meters
            base_distance = self.distance_to(destination_node)
            if base_distance <= 250:
                return False
            # --- Adaptive TTL based on neighbor count and TGNN suggestion ---
            neighbor_count = sum(1 for pkts in self.queue.values() for _ in pkts)
            # Prepare dummy features for TTL prediction
            nodes_features = [
                [self.node_id, self.x, self.y, self.z, self.energy, 1.0, 0, 0, 0, self.energy, 0, base_distance]
            ]
            edge_index = []
            edge_attr = []
            idx = 1
            for neighbor_ip, pkts in self.queue.items():
                for pkt in pkts:
                    neighbor = pkt.node
                    nodes_features.append([
                        neighbor.node_id, neighbor.x, neighbor.y, neighbor.z, neighbor.energy, 0.0, 0, 0, 0, neighbor.energy, 0, neighbor.distance_to(destination_node)
                    ])
                    edge_index.append([0, idx])
                    # Use dummy edge features for TTL prediction
                    edge_attr.append([0.0, 0.0, 0.0])
                    idx += 1
            # Use TGNN to predict TTL if enough neighbors, else fallback
            if len(nodes_features) > 1:
                tgnn_ttl = Node.tgnn_model.predict_ttl(nodes_features, edge_index, edge_attr)
                ttl = int(min(300, max(30, tgnn_ttl)))
            else:
                # Fallback: density-based TTL
                ttl = int(min(300, max(30, base_distance / 5 + neighbor_count * 2)))
            packet = self.create_packet(destination_ip, ttl, self.mac_address, self.x, self.y, self.z)
            if packet:
                Node.packet_sent_count += 1
                # Set creation_timestamp using simulation time
                packet.creation_timestamp = current_sim_time
                self.opp_packet_queue.append(packet)
                return True
        return False

    def create_packet(self, destination_ip, ttl, source_mac_address, source_x, source_y, source_z): # Added source_mac_address and position
        """Creates an OppPacket instance."""
        packet = OppPacket(
            source_ip=self.ip_address,
            destination_ip=destination_ip,
            ttl=ttl,
            source_mac_address=source_mac_address, # Pass the source MAC
            source_x=source_x,
            source_y=source_y,
            source_z=source_z
        )
        # Set creation_timestamp using simulation time (handled in send_packet)
        # packet.creation_timestamp = datetime.now()  # REMOVE
        return packet

    def process_queue(self):
        """Processes packets in the opportunistic packet queue using TGNN for forwarding."""
        processed_packets = []
        for _ in range(len(self.opp_packet_queue)):
            packet = self.opp_packet_queue.popleft()
            if packet.delivered:
                continue
            if packet.ttl <= 0:
                if not packet.delivered:
                    Node.drop_count += 1
                continue
            success = self.tgnn_forward(packet)
            if not success:
                packet.ttl -= 1
                if packet.ttl > 0:
                    processed_packets.append(packet)
                else:
                    if not packet.delivered:
                        Node.drop_count += 1
        self.opp_packet_queue.extend(processed_packets)

    def tgnn_forward(self, opp_packet):
        """Uses GNN-based neighbor scoring for next-hop selection (progress-only)."""
        current_node = self
        opp_packet.current_hop_mac = current_node.mac_address

        # --- Simulate channel noise: randomly drop packet with probability NOISE_LEVEL ---
        if NOISE_LEVEL > 0.0 and random.random() < NOISE_LEVEL:
            Node.drop_count += 1
            return False  # Packet dropped due to noise

        # 1. Check if destination
        if current_node.ip_address == opp_packet.destination_ip:
            if hasattr(opp_packet, "delivered") and opp_packet.delivered:
                return True
            opp_packet.delivered = True
            Node.packet_delivered_count += 1
            # Set delivery_timestamp using simulation time
            opp_packet.delivery_timestamp = current_sim_time
            print(f"Packet delivered! Source: {opp_packet.source_ip}, Dest: {opp_packet.destination_ip}, Created: {opp_packet.creation_timestamp}, Delivered at: {opp_packet.delivery_timestamp}")

            Node.delivered_packets.append({
                'initial_ttl': opp_packet.initial_ttl,
                'final_ttl': opp_packet.ttl,
                'hops_used': opp_packet.initial_ttl - opp_packet.ttl,
                'creation_timestamp': opp_packet.creation_timestamp,
                'delivery_timestamp': opp_packet.delivery_timestamp
            })

            packet_size = sys.getsizeof(opp_packet) + sum(sys.getsizeof(attr_value) for attr_value in opp_packet.__dict__.values())
            Node.delivered_packet_sizes.append(packet_size)

            current_node.opp_dest_packet.append(opp_packet)
            current_node.packets = np.append(current_node.packets, opp_packet)

            # --- Step 3: Create and send distance update back to source ---
            distance_back_to_source = math.sqrt(
                (current_node.x - opp_packet.source_x)**2 +
                (current_node.y - opp_packet.source_y)**2 +
                (current_node.z - opp_packet.source_z)**2
            )

            update_pkt = DistanceUpdatePacket(
                source_ip=current_node.ip_address,
                destination_ip=opp_packet.source_ip,
                distance=distance_back_to_source,
                visited_nodes_bloom_filter=None, # Pass None, not used
                original_opp_destination_ip=opp_packet.destination_ip
            )
            current_node.send_distance_update(update_pkt)

            return True # Packet delivered

        # 2. Check if TTL expired
        if opp_packet.ttl <= 0:
            Node.drop_count += 1
            return False # Packet dropped

        # 3. Gather neighbor info for GNN input
        self.gnn_agent.update_neighbors()
        dst_node = next((n for n in config.nodes if n.ip_address == opp_packet.destination_ip), None)
        if not self.gnn_agent.neighbors or dst_node is None:
            Node.drop_count += 1
            return False

        # Only consider neighbors that make progress toward the destination
        next_node = self.gnn_agent.select_best_forwarder(dst_node)
        if next_node and next_node.energy > 0 and self.energy > 0:
            self.energy = max(0, self.energy - 0.0003)
            next_node.energy = max(0, next_node.energy - 0.0001)
            next_node.opp_packet_queue.append(opp_packet)
            # Optionally: online GNN update (reward = 1 for delivery, 0 for not delivered)
            # Feedback can be shaped as in reference if desired
            return True
        else:
            Node.drop_count += 1
            return False

# === Simulation Configuration and Execution ---
class MobilityModel:
    def __init__(self, config, space_dim, speed_range=(1, 50)):
        self.config = config
        self.space_dim = space_dim
        self.speed_range = speed_range
        self.node_targets = {node.node_id: self.set_new_target(node) for node in self.config.nodes}

    def set_new_target(self, node):
        target_x = random.uniform(0, self.space_dim)
        target_y = random.uniform(0, self.space_dim)
        target_z = random.uniform(0, self.space_dim)
        speed = random.uniform(*self.speed_range)
        return (target_x, target_y, target_z, speed)

    def move_node(self, node):
        target_x, target_y, target_z, speed = self.node_targets[node.node_id]
        distance = math.sqrt((target_x - node.x) ** 2 + (target_y - node.y) ** 2 + (target_z - node.z) ** 2)

        if distance < speed:
            node.x, node.y, node.z = target_x, target_y, target_z
            node.position = np.append(node.position, np.array([{'x': node.x, 'y': node.y, 'z': node.z, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]))
            node.sort_mobility()
            self.node_targets[node.node_id] = self.set_new_target(node)
        else:
            direction_x = (target_x - node.x) / distance
            direction_y = (target_y - node.y) / distance
            direction_z = (target_z - node.z) / distance
            node.x += speed * direction_x
            node.y += speed * direction_y
            node.z += speed * direction_z
            node.position = np.append(node.position, np.array([{'x': node.x, 'y': node.y, 'z': node.z, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]))
            node.sort_mobility()

    def update_positions(self):
        for node in self.config.nodes:
            self.move_node(node)

class Configure:
    # Modify the constructor to accept num_nodes
    def __init__(self, num_nodes, space_dim=1000):
        self.space_dim = space_dim
        self.nodes = []
        self.existing_positions = set()
        # Initialize nodes list with the specified number of nodes
        for i in range(num_nodes):
            ip_address = f"192.168.1.{i + 1}"
            mac_address = f"00:0a:95:9d:68:{i + 1:02x}"
            x = round(random.uniform(0, self.space_dim), 2)
            y = round(random.uniform(0, self.space_dim), 2)
            z = round(random.uniform(0, self.space_dim), 2)
            # Pass i as a simple node_id for now, or use generate_unique_node_id if needed
            node = Node(ip_address, mac_address, x, y, z, node_id=i)
            self.nodes.append(node)
        # Assign GNN agent to each node after all nodes are created
        for node in self.nodes:
            node.gnn_agent = GNNForwardAgent(node, self.nodes)

    def generate_unique_position(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            x = round(random.uniform(0, self.space_dim), 2)
            y = round(random.uniform(0, self.space_dim), 2)
            z = round(random.uniform(0, self.space_dim), 2)
            pos_key = (x, y, z)
            if pos_key not in self.existing_positions:
                self.existing_positions.add(pos_key)
                return x, y, z
        raise RuntimeError("Unable to generate unique position")


class Opportunistic:
    def __init__(self, config):
        self.config = config

    def forwarder(self):
        for i in range(len(self.config.nodes)):
            for j in range(len(self.config.nodes)):
                if i != j:
                    node_i = self.config.nodes[i]
                    node_j = self.config.nodes[j]
                    # Slightly increase communication range for better connectivity
                    distance = node_i.distance_to(node_j)
                    if distance <= 300:  # Increased from 250m to 300m
                        # node_i discovers node_j
                        hello_pkt_i = node_j.hello(
                            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            distance=distance,
                            node=node_j
                        )
                        if hello_pkt_i:
                            node_i.receive_hello_packet(hello_pkt_i)

                        # node_j discovers node_i (bi-directional discovery)
                        hello_pkt_j = node_i.hello(
                            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            distance=distance,
                            node=node_i
                        )
                        if hello_pkt_j:
                            node_j.receive_hello_packet(hello_pkt_j)


# Define the node counts and speeds to simulate
num_nodes_list = list(range(100, 600, 100))  # 100, 200, 300, 400, 500
speeds_to_simulate =  [20]#list(range(20, 41, 5))  # 20, 25, 30, 35, 40

# --- Add this line to define noise_levels ---
noise_levels = [0.0,0.05,0.1,0.15,0.2]  # You can add more values, e.g., [0.0, 0.05, 0.1, 0.2]

# --- Update all metrics dictionaries to use (num_nodes, speed, noise_level, src_dst_distance) as key ---
pdr_results = {}
e2e_delay_stats_results = {}
throughput_results = {}
avg_reward_results = {}
avg_success_rate_results = {}
avg_on_bits_stats_results = {}
min_energy_results = {}
max_hops_results = {}
avg_initial_ttl_results = {}
avg_final_ttl_results = {}
simulation_duration_results = {}

# Path for the CSV file to store results
csv_results_path = "simulation_results.csv"

# Write CSV header if file does not exist or is empty
if not os.path.exists(csv_results_path) or os.path.getsize(csv_results_path) == 0:
    with open(csv_results_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Num Nodes", "Speed (m/s)", "Noise Level", "Src-Dst Distance (m)", "PDR (%)", "E2E Delay Mean (s)", "E2E Delay Median (s)", "E2E Delay Std (s)",
            "E2E Delay Min (s)", "E2E Delay Max (s)", "Estimated Throughput (bps)",
            "Avg Reward Closer (+1)", "Avg Reward Stationary (0)", "Avg Reward Away (-1)",
            "Avg Success Rate Closer (+1)", "Avg Success Rate Stationary (0)", "Avg Success Rate Away (-1)",
            "Min Remaining Energy (%)", "Max Hops Used", "Avg Initial TTL", "Avg Final TTL", "Simulation Duration (s)"
        ])

# --- Outer loop over noise levels ---
for noise_level in noise_levels:
    # global NOISE_LEVEL
    NOISE_LEVEL = noise_level
    print(f"\n=== Running Simulations with Noise Level {noise_level} ===")
    for num_nodes_sim in num_nodes_list:
        for speed_sim in speeds_to_simulate:
            print(f"\n--- Running Simulation with {num_nodes_sim} Nodes at Speed {speed_sim} m/s, Noise Level {noise_level} ---")

            # Reset static counters and lists for metrics
            Node.packet_sent_count = 0
            Node.packet_delivered_count = 0
            Node.drop_count = 0
            Node.delivered_packets = []
            Node.delivered_packet_sizes = []
            Node.delivered_visited_nodes_sizes = []

            # Setup network - pass num_nodes_sim to Configure
            config = Configure(num_nodes=num_nodes_sim)
            opportunistic = Opportunistic(config)
            mobility = MobilityModel(config, space_dim=1000, speed_range=(speed_sim, speed_sim))

            # --- Load GNN weights for all agents before simulation ---
            for node in config.nodes:
                if hasattr(node, "gnn_agent") and node.gnn_agent is not None:
                    node.gnn_agent.load_weights()

            # --- Select source and destination nodes with distance > 250m ---
            # Try all pairs until a valid pair is found
            found_pair = False
            for i in range(len(config.nodes)):
                for j in range(len(config.nodes)):
                    if i == j:
                        continue
                    src_candidate = config.nodes[i]
                    dst_candidate = config.nodes[j]
                    dist = src_candidate.distance_to(dst_candidate)
                    if dist > 250:
                        source = src_candidate
                        destination = dst_candidate
                        src_dst_distance = dist
                        found_pair = True
                        break
                if found_pair:
                    break
            if not found_pair:
                print("Warning: Could not find a source-destination pair with distance > 250m. Skipping this simulation.")
                continue

            # Define packet transmission configuration
            total_packets_to_send = 100 # Send only 300 packets
            packets_per_second = 20 # Increased packet sending rate
            sent_packet_count = 0
            simulation_steps = math.ceil(total_packets_to_send / packets_per_second) + 50  # more extra steps for late deliveries


            # --- Simulation Loop ---
            start_time = time.time()
            for timestep in range(simulation_steps):
                # print(f"--- Time step {timestep} ---") # Suppress detailed timestep logging

                # 1. Update node positions
                mobility.update_positions()

                # 2. Discover neighbors (Hello packets exchanged)
                opportunistic.forwarder()

                # 3. Send packets this step (if not all sent)
                if sent_packet_count < total_packets_to_send:
                    packets_this_step = min(packets_per_second, total_packets_to_send - sent_packet_count)
                    for _ in range(packets_this_step):
                        source.send_packet(destination.ip_address)
                    sent_packet_count += packets_this_step

                # 4. Node updates: position logging, queue processing, and distance update processing
                for node in config.nodes:
                    node.update_position_log()
                    node.process_queue() # Process OppPackets
                    node.process_distance_updates() # Process DistanceUpdatePackets

                # Optional: simulate real time
                # time.sleep(0.01)

                # --- Increment simulation time ---
                current_sim_time += SIM_TIME_STEP

                # Log intermediate results every 20 steps
                if timestep % 20 == 0 or timestep == simulation_steps - 1:
                    total_sent = Node.packet_sent_count
                    total_delivered = Node.packet_delivered_count
                    total_dropped = Node.drop_count
                    pdr = (total_delivered / total_sent) * 100 if total_sent > 0 else 0.0
                    print(f"[Step {timestep}] Sent: {total_sent}, Delivered: {total_delivered}, Dropped: {total_dropped}, PDR: {pdr:.2f}%")

            end_time = time.time()
            simulation_duration = end_time - start_time
            simulation_duration_results[(num_nodes_sim, speed_sim, noise_level)] = simulation_duration
            print(f"Simulation with {num_nodes_sim} nodes at speed {speed_sim} m/s, noise level {noise_level} finished in {simulation_duration:.2f} seconds.")

            # --- Calculate Performance Metrics ---

            # PDR
            total_packets_sent = Node.packet_sent_count
            total_packets_delivered = Node.packet_delivered_count
            pdr = (total_packets_delivered / total_packets_sent) * 100 if total_packets_sent > 0 else 0.0
            pdr_results[(num_nodes_sim, speed_sim, noise_level)] = pdr
            print(f"  PDR: {pdr:.2f}%")


            # End-to-End Delay
            end_to_end_delays = []
            if Node.delivered_packets:
                for packet_info in Node.delivered_packets:
                    if 'creation_timestamp' in packet_info and 'delivery_timestamp' in packet_info:
                        creation_time = packet_info['creation_timestamp']
                        delivery_time = packet_info['delivery_timestamp']
                        # Calculate delay using simulation time (float)
                        delay = delivery_time - creation_time
                        end_to_end_delays.append(delay)
            if end_to_end_delays:
                e2e_delay_stats = pd.Series(end_to_end_delays).describe()
                e2e_delay_stats_results[(num_nodes_sim, speed_sim, noise_level)] = e2e_delay_stats
                print(f"  Avg E2E Delay: {e2e_delay_stats['mean']:.2f} seconds")
            else:
                e2e_delay_stats_results[(num_nodes_sim, speed_sim, noise_level)] = None
                print("  Avg E2E Delay: N/A (no delivered packets)")


            # Estimated Throughput
            estimated_average_packet_size_bytes = 0
            if Node.delivered_packet_sizes:
                estimated_average_packet_size_bytes = np.mean(Node.delivered_packet_sizes)

            throughput_bps = 0
            if total_packets_delivered > 0 and simulation_duration > 0 and estimated_average_packet_size_bytes > 0:
                total_data_delivered_bytes = total_packets_delivered * estimated_average_packet_size_bytes
                total_data_delivered_bits = total_data_delivered_bytes * 8
                throughput_bps = total_data_delivered_bits / simulation_duration

            throughput_results[(num_nodes_sim, speed_sim, noise_level)] = throughput_bps
            print(f"  Estimated Throughput: {throughput_bps / 1000:.2f} Kbps")


            # Direction-Based Analysis (Aggregate from all nodes)
            total_reward_by_direction = defaultdict(list)
            success_rate_by_direction = defaultdict(list)

            for node in config.nodes:
                for direction, rewards in node.reward_by_direction.items():
                    if direction != -99:
                        total_reward_by_direction[direction].extend(rewards)

                for direction, outcome in node.success_by_direction.items():
                     if direction != -99 and outcome["total"] > 0:
                        success_rate = outcome["success"] / outcome["total"]
                        success_rate_by_direction[direction].append(success_rate)
                     elif direction != -99 and outcome["total"] == 0:
                         success_rate_by_direction[direction].append(0)

            # Calculate averages here
            avg_reward_by_direction = {d: np.mean(r) for d, r in total_reward_by_direction.items() if r}
            avg_success_rate_by_direction = {d: np.mean(rates) for d, rates in success_rate_by_direction.items() if rates}

            avg_reward_results[(num_nodes_sim, speed_sim, noise_level)] = avg_reward_by_direction
            avg_success_rate_results[(num_nodes_sim, speed_sim, noise_level)] = avg_success_rate_by_direction
            print(f"  Avg Reward by Direction: {avg_reward_by_direction}")
            print(f"  Avg Success Rate by Direction: {avg_success_rate_by_direction}")


            # Minimum Remaining Energy
            energies = [node.energy for node in config.nodes]
            min_energy = min(energies) if energies else 0
            min_energy_results[(num_nodes_sim, speed_sim, noise_level)] = min_energy
            print(f"  Minimum Remaining Energy: {min_energy:.2f}%")


            # Maximum Hops Used
            max_hops = 0
            if Node.delivered_packets:
                 hops_used = [p['hops_used'] for p in Node.delivered_packets if 'hops_used' in p]
                 max_hops = max(hops_used) if hops_used else 0
            max_hops_results[(num_nodes_sim, speed_sim, noise_level)] = max_hops
            print(f"  Maximum Hops Used: {max_hops}")


            # Average Initial and Final TTL
            avg_initial_ttl = 0
            avg_final_ttl = 0
            if Node.delivered_packets:
                initial_ttls = [p['initial_ttl'] for p in Node.delivered_packets if 'initial_ttl' in p]
                final_ttls = [p['final_ttl'] for p in Node.delivered_packets if 'final_ttl' in p]
                avg_initial_ttl = np.mean(initial_ttls) if initial_ttls else 0
                avg_final_ttl = np.mean(final_ttls) if final_ttls else 0
            avg_initial_ttl_results[(num_nodes_sim, speed_sim, noise_level)] = avg_initial_ttl
            avg_final_ttl_results[(num_nodes_sim, speed_sim, noise_level)] = avg_final_ttl
            print(f"  Avg Initial TTL: {avg_initial_ttl:.2f}")
            print(f"  Avg Final TTL: {avg_final_ttl:.2f}")

            # Store results using (num_nodes_sim, speed_sim, noise_level, src_dst_distance) as key, and also store src_dst_distance
            key = (num_nodes_sim, speed_sim, noise_level, src_dst_distance)
            # Use a tuple with noise_level and distance for all results
            pdr_results[key] = pdr
            e2e_delay_stats_results[key] = e2e_delay_stats if end_to_end_delays else None
            throughput_results[key] = throughput_bps
            avg_reward_results[key] = avg_reward_by_direction
            avg_success_rate_results[key] = avg_success_rate_by_direction
            min_energy_results[key] = min_energy
            max_hops_results[key] = max_hops
            avg_initial_ttl_results[key] = avg_initial_ttl
            avg_final_ttl_results[key] = avg_final_ttl
            simulation_duration_results[key] = simulation_duration

            # --- CSV append for this simulation ---
            with open(csv_results_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    num_nodes_sim,
                    speed_sim,
                    noise_level,
                    src_dst_distance,
                    pdr,
                    e2e_delay_stats['mean'] if end_to_end_delays else None,
                    e2e_delay_stats['50%'] if end_to_end_delays else None,
                    e2e_delay_stats['std'] if end_to_end_delays else None,
                    e2e_delay_stats['min'] if end_to_end_delays else None,
                    e2e_delay_stats['max'] if end_to_end_delays else None,
                    throughput_bps,
                    avg_reward_by_direction.get(1, None),
                    avg_reward_by_direction.get(0, None),
                    avg_reward_by_direction.get(-1, None),
                    avg_success_rate_by_direction.get(1, None),
                    avg_success_rate_by_direction.get(0, None),
                    avg_success_rate_by_direction.get(-1, None),
                    min_energy,
                    max_hops,
                    avg_initial_ttl,
                    avg_final_ttl,
                    simulation_duration
                ])

print("\n--- Simulation Runs Complete ---")

# Initialize a dictionary to hold the summarized metrics
performance_metrics_summary = {}

# Iterate through each (num_nodes, speed, noise_level, distance) tuple that was simulated
for key in pdr_results.keys():
    num_nodes, speed, noise_level, src_dst_distance = key
    metrics_for_key = {}

    metrics_for_key['Src-Dst Distance (m)'] = src_dst_distance
    metrics_for_key['PDR (%)'] = pdr_results.get(key)
    e2e_delay_stats = e2e_delay_stats_results.get(key)
    if e2e_delay_stats is not None:
        metrics_for_key['E2E Delay Mean (s)'] = e2e_delay_stats['mean']
        metrics_for_key['E2E Delay Median (s)'] = e2e_delay_stats['50%']
        metrics_for_key['E2E Delay Std (s)'] = e2e_delay_stats['std']
        metrics_for_key['E2E Delay Min (s)'] = e2e_delay_stats['min']
        metrics_for_key['E2E Delay Max (s)'] = e2e_delay_stats['max']
    else:
        metrics_for_key['E2E Delay Mean (s)'] = None
        metrics_for_key['E2E Delay Median (s)'] = None
        metrics_for_key['E2E Delay Std (s)'] = None
        metrics_for_key['E2E Delay Min (s)'] = None
        metrics_for_key['E2E Delay Max (s)'] = None

    metrics_for_key['Estimated Throughput (bps)'] = throughput_results.get(key)
    avg_rewards = avg_reward_results.get(key, {})
    metrics_for_key['Avg Reward Closer (+1)'] = avg_rewards.get(1, None)
    metrics_for_key['Avg Reward Stationary (0)'] = avg_rewards.get(0, None)
    metrics_for_key['Avg Reward Away (-1)'] = avg_rewards.get(-1, None)
    avg_success_rates = avg_success_rate_results.get(key, {})
    metrics_for_key['Avg Success Rate Closer (+1)'] = avg_success_rates.get(1, None)
    metrics_for_key['Avg Success Rate Stationary (0)'] = avg_success_rates.get(0, None)
    metrics_for_key['Avg Success Rate Away (-1)'] = avg_success_rates.get(-1, None)
    metrics_for_key['Min Remaining Energy (%)'] = min_energy_results.get(key)
    metrics_for_key['Max Hops Used'] = max_hops_results.get(key)
    metrics_for_key['Avg Initial TTL'] = avg_initial_ttl_results.get(key)
    metrics_for_key['Avg Final TTL'] = avg_final_ttl_results.get(key)
    metrics_for_key['Simulation Duration (s)'] = simulation_duration_results.get(key)
    metrics_for_key['Noise Level'] = noise_level

    # Use tuple (num_nodes, speed, noise_level, src_dst_distance) as index
    performance_metrics_summary[key] = metrics_for_key

# Convert the summary dictionary to a Pandas DataFrame for better visualization and analysis
performance_summary_df = pd.DataFrame.from_dict(performance_metrics_summary, orient='index')
performance_summary_df.index.names = ['Num Nodes', 'Speed (m/s)', 'Noise Level', 'Src-Dst Distance (m)']

# Save the summary DataFrame to CSV (append, for full summary)
performance_summary_df.to_csv(csv_results_path, mode='a', header=True)

# Display the summary DataFrame
print("\n=== Performance Metrics Summary Across Speeds and Noise Levels ===")
print(performance_summary_df)

# Print all metrics for each speed and noise level in a readable format
print("\n=== Detailed Metrics for Each Speed and Noise Level ===")
for idx in performance_summary_df.index:
    print(f"\n--- Metrics for Num Nodes {idx[0]}, Speed {idx[1]} m/s, Noise Level {idx[2]} ---")
    for metric, value in performance_summary_df.loc[idx].items():
        print(f"{metric}: {value}")

# Metrics to visualize as line plots
line_plot_metrics = [
    'PDR (%)',
    'E2E Delay Mean (s)',
    'Estimated Throughput (bps)',
    'Min Remaining Energy (%)',
    'Max Hops Used',
    'Avg Initial TTL',
    'Avg Final TTL',
    'Simulation Duration (s)'
]

# Metrics to visualize as bar plots (direction-based)
bar_plot_direction_metrics_reward = [
    'Avg Reward Closer (+1)',
    'Avg Reward Stationary (0)',
    'Avg Reward Away (-1)'
]

bar_plot_direction_metrics_success = [
    'Avg Success Rate Closer (+1)',
    'Avg Success Rate Stationary (0)',
    'Avg Success Rate Away (-1)'
]

# --- Show individual graphs/charts for each metric, grouped by noise level ---

# 1. Line plots for each line metric, grouped by noise level
print("\nGenerating Individual Line Plots for Each Metric (grouped by Noise Level):")
for metric in line_plot_metrics:
    if metric in performance_summary_df.columns:
        plt.figure(figsize=(8, 5))
        for noise_level in noise_levels:
            # Filter rows for this noise level
            df_noise = performance_summary_df[performance_summary_df['Noise Level'] == noise_level]
            # Use Num Nodes or Speed as x-axis (choose one, here use Num Nodes)
            x = df_noise.index.get_level_values('Num Nodes')
            y = df_noise[metric].replace({None: np.nan})
            plt.plot(x, y, marker='o', linestyle='-', label=f"Noise {noise_level}")
        plt.xlabel("Num Nodes")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Num Nodes (by Noise Level)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metric '{metric}' not found in performance_summary_df.")

# 2. Bar plots for each direction-based reward metric, grouped by noise level
print("\nGenerating Individual Bar Plots for Direction-Based Reward Metrics (grouped by Noise Level):")
for metric in bar_plot_direction_metrics_reward:
    if metric in performance_summary_df.columns:
        plt.figure(figsize=(8, 5))
        width = 0.18
        x_vals = sorted(set(performance_summary_df.index.get_level_values('Num Nodes')))
        for i, noise_level in enumerate(noise_levels):
            df_noise = performance_summary_df[performance_summary_df['Noise Level'] == noise_level]
            y = [df_noise[df_noise.index.get_level_values('Num Nodes') == x][metric].mean() for x in x_vals]
            plt.bar([x + i*width for x in range(len(x_vals))], y, width=width, label=f"Noise {noise_level}")
        plt.xlabel("Num Nodes")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Num Nodes (by Noise Level)")
        plt.grid(axis='y', alpha=0.75)
        plt.xticks([x + width*(len(noise_levels)-1)/2 for x in range(len(x_vals))], x_vals)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metric '{metric}' not found in performance_summary_df.")

# 3. Bar plots for each direction-based success rate metric, grouped by noise level
print("\nGenerating Individual Bar Plots for Direction-Based Success Rate Metrics (grouped by Noise Level):")
for metric in bar_plot_direction_metrics_success:
    if metric in performance_summary_df.columns:
        plt.figure(figsize=(8, 5))
        width = 0.18
        x_vals = sorted(set(performance_summary_df.index.get_level_values('Num Nodes')))
        for i, noise_level in enumerate(noise_levels):
            df_noise = performance_summary_df[performance_summary_df['Noise Level'] == noise_level]
            y = [df_noise[df_noise.index.get_level_values('Num Nodes') == x][metric].mean() for x in x_vals]
            plt.bar([x + i*width for x in range(len(x_vals))], y, width=width, label=f"Noise {noise_level}")
        plt.xlabel("Num Nodes")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Num Nodes (by Noise Level)")
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.75)
        plt.xticks([x + width*(len(noise_levels)-1)/2 for x in range(len(x_vals))], x_vals)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metric '{metric}' not found in performance_summary_df.")

# Save the GNN weights after simulation
GNN_WEIGHTS_PATH = "gnn_forward_agent_weights.pt"
for node in config.nodes:
    if hasattr(node, "gnn_agent") and node.gnn_agent is not None:
        node.gnn_agent.save_weights()