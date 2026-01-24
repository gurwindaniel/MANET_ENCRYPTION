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
from datetime import datetime
import matplotlib.pyplot as plt
import zlib # Import zlib for CRC32
import mmh3 # Import MurmurHash3 for better hashing
import sys # Import sys for object size analysis
# import tgnn # Import TGNN for graph neural network-based forwarding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import NNConv, GCNConv

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

# === Bloom Filter Implementation ===
class BloomFilter:
    def __init__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate

        m_optimal = int(-(capacity * math.log(error_rate)) / (math.log(2)**2))
        k_optimal = int((m_optimal / capacity) * math.log(2))

        self.m = min(max(m_optimal, 256), 1024) # Increase bit array for lower false positive
        self.k = max(4, int((self.m / capacity) * math.log(2))) # Increase hash functions

        self.m = (self.m + 7) // 8 * 8
        self.bit_array = bytearray(self.m // 8)

    def _hash(self, element, seed):
        """Compute a hash value for the element using MurmurHash3."""
        # Using MurmurHash3 because it's fast and has good distribution
        # Ensure element is hashable (e.g., convert to string)
        return mmh3.hash(str(element), seed) % self.m

    def add(self, element):
        """Add an element to the bloom filter."""
        for i in range(self.k):
            index = self._hash(element, i)
            byte_index = index // 8
            bit_offset = index % 8
            self.bit_array[byte_index] |= (1 << bit_offset)

    def might_contain(self, element):
        """Check if an element might be in the bloom filter."""
        for i in range(self.k):
            index = self._hash(element, i)
            byte_index = index // 8
            bit_offset = index % 8
            if not (self.bit_array[byte_index] & (1 << bit_offset)):
                return False # Definitely not in the set
        return True # Possibly in the set (could be a false positive)

# === Packet Classes ===
class OppPacket:
    def __init__(self, source_ip, destination_ip, ttl, source_mac_address, source_x, source_y, source_z): # Added source_mac_address and position
        self.source_ip = source_ip # Original source IP
        self.destination_ip = destination_ip
        self.ttl = ttl
        # Add creation timestamp
        self.creation_timestamp = datetime.now()
        self.visited_nodes = BloomFilter(capacity=100, error_rate=0.01)
        self.initial_ttl = ttl # Store initial TTL for analysis
        self.delivered = False # Track if the packet has been delivered
        self.current_hop_mac = source_mac_address # Store the MAC of the current node holding/forwarding the packet
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
        self.source_ip = source_ip # The node sending the update (likely the destination or an intermediate node)
        self.destination_ip = destination_ip # The original source node of the OppPacket
        self.distance = distance # The distance from the source_ip of this update packet to the original destination
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.visited_nodes = visited_nodes_bloom_filter # Carry the bloom filter for the RETURN path
        self.original_opp_destination_ip = original_opp_destination_ip # Add original destination IP
        self.ttl = 50 # Add TTL for distance update packets (arbitrary initial value)


# === Node Class ===
class Node:
    packet_sent_count = 0
    packet_delivered_count = 0
    drop_count = 0
    delivered_packets = [] # List to store info about delivered packets
    # Add lists to store packet and visited_nodes sizes for delivered packets
    delivered_packet_sizes = []
    delivered_visited_nodes_sizes = []


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
            # --- Adaptive TTL based on neighbor count and TGNN suggestion ---
            neighbor_count = sum(1 for pkts in self.queue.values() for _ in pkts)
            base_distance = self.distance_to(destination_node)
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
        # Explicitly set the creation_timestamp when the packet is created
        packet.creation_timestamp = datetime.now()
        # visited_nodes and other attributes are initialized in OppPacket.__init__
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
        opp_packet.visited_nodes.add(current_node.mac_address)
        opp_packet.current_hop_mac = current_node.mac_address

        # 1. Check if destination
        if current_node.ip_address == opp_packet.destination_ip:
            if hasattr(opp_packet, "delivered") and opp_packet.delivered:
                return True
            opp_packet.delivered = True
            Node.packet_delivered_count += 1
            print(f"Packet delivered! Source: {opp_packet.source_ip}, Dest: {opp_packet.destination_ip}, Created: {opp_packet.creation_timestamp}, Delivered at: {datetime.now()}")

            # Collect visited_nodes size at destination
            visited_nodes_size = 0
            if hasattr(opp_packet, 'visited_nodes') and isinstance(opp_packet.visited_nodes, BloomFilter):
                 visited_nodes_size = sys.getsizeof(opp_packet.visited_nodes.bit_array) + sys.getsizeof(opp_packet.visited_nodes)

            # Log delivered packet info
            Node.delivered_packets.append({
                'initial_ttl': opp_packet.initial_ttl,
                'final_ttl': opp_packet.ttl,
                'hops_used': opp_packet.initial_ttl - opp_packet.ttl,
                'creation_timestamp': opp_packet.creation_timestamp, # Log creation timestamp
                'delivery_timestamp': datetime.now(), # Log delivery timestamp
                'visited_nodes_size': visited_nodes_size # Include visited nodes size
            })

            # Collect packet size at destination (Python object size)
            packet_size = sys.getsizeof(opp_packet) + sum(sys.getsizeof(attr_value) for attr_value in opp_packet.__dict__.values())
            Node.delivered_packet_sizes.append(packet_size)
            # Collect visited_nodes size at destination (Python object size)
            Node.delivered_visited_nodes_sizes.append(visited_nodes_size)


            current_node.opp_dest_packet.append(opp_packet) # Store delivered packet at destination
            current_node.packets = np.append(current_node.packets, opp_packet) # Still keep in packets for history


            # --- Step 3: Create and send distance update back to source using a NEW bloom filter ---
            # Calculate distance back to source using position from OppPacket
            distance_back_to_source = math.sqrt(
                (current_node.x - opp_packet.source_x)**2 +
                (current_node.y - opp_packet.source_y)**2 +
                (current_node.z - opp_packet.source_z)**2
            )

            # Create a NEW bloom filter for the distance update packet's return journey
            update_bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
            # Add the current node (destination) to the update packet's bloom filter
            update_bloom_filter.add(current_node.mac_address)


            update_pkt = DistanceUpdatePacket(
                source_ip=current_node.ip_address, # Destination is the source of the update
                destination_ip=opp_packet.source_ip, # Send back to original source
                distance=distance_back_to_source, # Use calculated distance back to source
                visited_nodes_bloom_filter=update_bloom_filter, # Include the NEW visited nodes bloom filter for the return path
                original_opp_destination_ip=opp_packet.destination_ip # Include original destination IP
            )
            current_node.send_distance_update(update_pkt)
            # ------------------------------------------------------------------------------------------

            # --- Online TGNN reward: positive reward for delivery ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, 2.0, None, True  # Stronger reward for delivery
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
            return True # Packet delivered

        # 2. Check if TTL expired
        if opp_packet.ttl <= 0:
            Node.drop_count += 1
            # --- Online TGNN reward: negative reward for drop ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, -2.0, None, True  # Stronger penalty for drop
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
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

# === Bloom Filter Implementation ===
class BloomFilter:
    def __init__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate

        m_optimal = int(-(capacity * math.log(error_rate)) / (math.log(2)**2))
        k_optimal = int((m_optimal / capacity) * math.log(2))

        self.m = min(max(m_optimal, 256), 1024) # Increase bit array for lower false positive
        self.k = max(4, int((self.m / capacity) * math.log(2))) # Increase hash functions

        self.m = (self.m + 7) // 8 * 8
        self.bit_array = bytearray(self.m // 8)

    def _hash(self, element, seed):
        """Compute a hash value for the element using MurmurHash3."""
        # Using MurmurHash3 because it's fast and has good distribution
        # Ensure element is hashable (e.g., convert to string)
        return mmh3.hash(str(element), seed) % self.m

    def add(self, element):
        """Add an element to the bloom filter."""
        for i in range(self.k):
            index = self._hash(element, i)
            byte_index = index // 8
            bit_offset = index % 8
            self.bit_array[byte_index] |= (1 << bit_offset)

    def might_contain(self, element):
        """Check if an element might be in the bloom filter."""
        for i in range(self.k):
            index = self._hash(element, i)
            byte_index = index // 8
            bit_offset = index % 8
            if not (self.bit_array[byte_index] & (1 << bit_offset)):
                return False # Definitely not in the set
        return True # Possibly in the set (could be a false positive)

# === Packet Classes ===
class OppPacket:
    def __init__(self, source_ip, destination_ip, ttl, source_mac_address, source_x, source_y, source_z): # Added source_mac_address and position
        self.source_ip = source_ip # Original source IP
        self.destination_ip = destination_ip
        self.ttl = ttl
        # Add creation timestamp
        self.creation_timestamp = datetime.now()
        self.visited_nodes = BloomFilter(capacity=100, error_rate=0.01)
        self.initial_ttl = ttl # Store initial TTL for analysis
        self.delivered = False # Track if the packet has been delivered
        self.current_hop_mac = source_mac_address # Store the MAC of the current node holding/forwarding the packet
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
        self.source_ip = source_ip # The node sending the update (likely the destination or an intermediate node)
        self.destination_ip = destination_ip # The original source node of the OppPacket
        self.distance = distance # The distance from the source_ip of this update packet to the original destination
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.visited_nodes = visited_nodes_bloom_filter # Carry the bloom filter for the RETURN path
        self.original_opp_destination_ip = original_opp_destination_ip # Add original destination IP
        self.ttl = 50 # Add TTL for distance update packets (arbitrary initial value)


# === Node Class ===
class Node:
    packet_sent_count = 0
    packet_delivered_count = 0
    drop_count = 0
    delivered_packets = [] # List to store info about delivered packets
    # Add lists to store packet and visited_nodes sizes for delivered packets
    delivered_packet_sizes = []
    delivered_visited_nodes_sizes = []


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
            # --- Adaptive TTL based on neighbor count and TGNN suggestion ---
            neighbor_count = sum(1 for pkts in self.queue.values() for _ in pkts)
            base_distance = self.distance_to(destination_node)
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
        # Explicitly set the creation_timestamp when the packet is created
        packet.creation_timestamp = datetime.now()
        # visited_nodes and other attributes are initialized in OppPacket.__init__
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
        opp_packet.visited_nodes.add(current_node.mac_address)
        opp_packet.current_hop_mac = current_node.mac_address

        # 1. Check if destination
        if current_node.ip_address == opp_packet.destination_ip:
            if hasattr(opp_packet, "delivered") and opp_packet.delivered:
                return True
            opp_packet.delivered = True
            Node.packet_delivered_count += 1
            print(f"Packet delivered! Source: {opp_packet.source_ip}, Dest: {opp_packet.destination_ip}, Created: {opp_packet.creation_timestamp}, Delivered at: {datetime.now()}")

            # Collect visited_nodes size at destination
            visited_nodes_size = 0
            if hasattr(opp_packet, 'visited_nodes') and isinstance(opp_packet.visited_nodes, BloomFilter):
                 visited_nodes_size = sys.getsizeof(opp_packet.visited_nodes.bit_array) + sys.getsizeof(opp_packet.visited_nodes)

            # Log delivered packet info
            Node.delivered_packets.append({
                'initial_ttl': opp_packet.initial_ttl,
                'final_ttl': opp_packet.ttl,
                'hops_used': opp_packet.initial_ttl - opp_packet.ttl,
                'creation_timestamp': opp_packet.creation_timestamp, # Log creation timestamp
                'delivery_timestamp': datetime.now(), # Log delivery timestamp
                'visited_nodes_size': visited_nodes_size # Include visited nodes size
            })

            # Collect packet size at destination (Python object size)
            packet_size = sys.getsizeof(opp_packet) + sum(sys.getsizeof(attr_value) for attr_value in opp_packet.__dict__.values())
            Node.delivered_packet_sizes.append(packet_size)
            # Collect visited_nodes size at destination (Python object size)
            Node.delivered_visited_nodes_sizes.append(visited_nodes_size)


            current_node.opp_dest_packet.append(opp_packet) # Store delivered packet at destination
            current_node.packets = np.append(current_node.packets, opp_packet) # Still keep in packets for history


            # --- Step 3: Create and send distance update back to source using a NEW bloom filter ---
            # Calculate distance back to source using position from OppPacket
            distance_back_to_source = math.sqrt(
                (current_node.x - opp_packet.source_x)**2 +
                (current_node.y - opp_packet.source_y)**2 +
                (current_node.z - opp_packet.source_z)**2
            )

            # Create a NEW bloom filter for the distance update packet's return journey
            update_bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
            # Add the current node (destination) to the update packet's bloom filter
            update_bloom_filter.add(current_node.mac_address)


            update_pkt = DistanceUpdatePacket(
                source_ip=current_node.ip_address, # Destination is the source of the update
                destination_ip=opp_packet.source_ip, # Send back to original source
                distance=distance_back_to_source, # Use calculated distance back to source
                visited_nodes_bloom_filter=update_bloom_filter, # Include the NEW visited nodes bloom filter for the return path
                original_opp_destination_ip=opp_packet.destination_ip # Include original destination IP
            )
            current_node.send_distance_update(update_pkt)
            # ------------------------------------------------------------------------------------------

            # --- Online TGNN reward: positive reward for delivery ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, 2.0, None, True  # Stronger reward for delivery
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
            return True # Packet delivered

        # 2. Check if TTL expired
        if opp_packet.ttl <= 0:
            Node.drop_count += 1
            # --- Online TGNN reward: negative reward for drop ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, -2.0, None, True  # Stronger penalty for drop
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
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

# === Patch Node to use GNNForwardAgent for forwarding ===
class Node:
    packet_sent_count = 0
    packet_delivered_count = 0
    drop_count = 0
    delivered_packets = [] # List to store info about delivered packets
    # Add lists to store packet and visited_nodes sizes for delivered packets
    delivered_packet_sizes = []
    delivered_visited_nodes_sizes = []


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
            # --- Adaptive TTL based on neighbor count and TGNN suggestion ---
            neighbor_count = sum(1 for pkts in self.queue.values() for _ in pkts)
            base_distance = self.distance_to(destination_node)
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
        # Explicitly set the creation_timestamp when the packet is created
        packet.creation_timestamp = datetime.now()
        # visited_nodes and other attributes are initialized in OppPacket.__init__
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
        opp_packet.visited_nodes.add(current_node.mac_address)
        opp_packet.current_hop_mac = current_node.mac_address

        # 1. Check if destination
        if current_node.ip_address == opp_packet.destination_ip:
            if hasattr(opp_packet, "delivered") and opp_packet.delivered:
                return True
            opp_packet.delivered = True
            Node.packet_delivered_count += 1
            print(f"Packet delivered! Source: {opp_packet.source_ip}, Dest: {opp_packet.destination_ip}, Created: {opp_packet.creation_timestamp}, Delivered at: {datetime.now()}")

            # Collect visited_nodes size at destination
            visited_nodes_size = 0
            if hasattr(opp_packet, 'visited_nodes') and isinstance(opp_packet.visited_nodes, BloomFilter):
                 visited_nodes_size = sys.getsizeof(opp_packet.visited_nodes.bit_array) + sys.getsizeof(opp_packet.visited_nodes)

            # Log delivered packet info
            Node.delivered_packets.append({
                'initial_ttl': opp_packet.initial_ttl,
                'final_ttl': opp_packet.ttl,
                'hops_used': opp_packet.initial_ttl - opp_packet.ttl,
                'creation_timestamp': opp_packet.creation_timestamp, # Log creation timestamp
                'delivery_timestamp': datetime.now(), # Log delivery timestamp
                'visited_nodes_size': visited_nodes_size # Include visited nodes size
            })

            # Collect packet size at destination (Python object size)
            packet_size = sys.getsizeof(opp_packet) + sum(sys.getsizeof(attr_value) for attr_value in opp_packet.__dict__.values())
            Node.delivered_packet_sizes.append(packet_size)
            # Collect visited_nodes size at destination (Python object size)
            Node.delivered_visited_nodes_sizes.append(visited_nodes_size)


            current_node.opp_dest_packet.append(opp_packet) # Store delivered packet at destination
            current_node.packets = np.append(current_node.packets, opp_packet) # Still keep in packets for history


            # --- Step 3: Create and send distance update back to source using a NEW bloom filter ---
            # Calculate distance back to source using position from OppPacket
            distance_back_to_source = math.sqrt(
                (current_node.x - opp_packet.source_x)**2 +
                (current_node.y - opp_packet.source_y)**2 +
                (current_node.z - opp_packet.source_z)**2
            )

            # Create a NEW bloom filter for the distance update packet's return journey
            update_bloom_filter = BloomFilter(capacity=100, error_rate=0.01)
            # Add the current node (destination) to the update packet's bloom filter
            update_bloom_filter.add(current_node.mac_address)


            update_pkt = DistanceUpdatePacket(
                source_ip=current_node.ip_address, # Destination is the source of the update
                destination_ip=opp_packet.source_ip, # Send back to original source
                distance=distance_back_to_source, # Use calculated distance back to source
                visited_nodes_bloom_filter=update_bloom_filter, # Include the NEW visited nodes bloom filter for the return path
                original_opp_destination_ip=opp_packet.destination_ip # Include original destination IP
            )
            current_node.send_distance_update(update_pkt)
            # ------------------------------------------------------------------------------------------

            # --- Online TGNN reward: positive reward for delivery ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, 2.0, None, True  # Stronger reward for delivery
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
            return True # Packet delivered

        # 2. Check if TTL expired
        if opp_packet.ttl <= 0:
            Node.drop_count += 1
            # --- Online TGNN reward: negative reward for drop ---
            if self.last_tgnn_state is not None and self.last_tgnn_action is not None:
                Node.tgnn_model.store_experience(
                    self.last_tgnn_state, self.last_tgnn_action, -2.0, None, True  # Stronger penalty for drop
                )
                self.last_tgnn_state = None
                self.last_tgnn_action = None
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


# Define the node count to simulate
num_nodes_sim = 100

# Define the speeds to simulate
# speeds_to_simulate = [40]  # Increased speed for better performance
speeds_to_simulate = list(range(20, 41, 5))  # 20, 25, 30, 35, 40

# Initialize dictionaries to store results for each speed
pdr_results = {}
e2e_delay_stats_results = {} # Store describe() output
throughput_results = {}
avg_reward_results = {}
avg_success_rate_results = {}
avg_on_bits_stats_results = {} # Store describe() output
min_energy_results = {}
max_hops_results = {}
avg_initial_ttl_results = {}
avg_final_ttl_results = {}
simulation_duration_results = {}


for speed_sim in speeds_to_simulate:
    print(f"\n--- Running Simulation with {num_nodes_sim} Nodes at Speed {speed_sim} m/s ---")

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
    # Set speed range
    mobility = MobilityModel(config, space_dim=1000, speed_range=(speed_sim, speed_sim))


    # Define source and destination nodes
    source = config.nodes[0]
    # Adjust the destination node index
    destination = config.nodes[num_nodes_sim - 1]


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

        # Log intermediate results every 20 steps
        if timestep % 20 == 0 or timestep == simulation_steps - 1:
            total_sent = Node.packet_sent_count
            total_delivered = Node.packet_delivered_count
            total_dropped = Node.drop_count
            pdr = (total_delivered / total_sent) * 100 if total_sent > 0 else 0.0
            print(f"[Step {timestep}] Sent: {total_sent}, Delivered: {total_delivered}, Dropped: {total_dropped}, PDR: {pdr:.2f}%")

    end_time = time.time()
    simulation_duration = end_time - start_time
    simulation_duration_results[speed_sim] = simulation_duration
    print(f"Simulation with {num_nodes_sim} nodes at speed {speed_sim} m/s finished in {simulation_duration:.2f} seconds.")

    # --- Calculate Performance Metrics ---

    # PDR
    total_packets_sent = Node.packet_sent_count
    total_packets_delivered = Node.packet_delivered_count
    pdr = (total_packets_delivered / total_packets_sent) * 100 if total_packets_sent > 0 else 0.0
    pdr_results[speed_sim] = pdr
    print(f"  PDR: {pdr:.2f}%")


    # End-to-End Delay
    end_to_end_delays = []
    if Node.delivered_packets:
        for packet_info in Node.delivered_packets:
            if 'creation_timestamp' in packet_info and 'delivery_timestamp' in packet_info:
                creation_time = packet_info['creation_timestamp']
                delivery_time = packet_info['delivery_timestamp']
                delay = (delivery_time - creation_time).total_seconds()
                end_to_end_delays.append(delay)
    if end_to_end_delays:
        e2e_delay_stats = pd.Series(end_to_end_delays).describe()
        e2e_delay_stats_results[speed_sim] = e2e_delay_stats
        print(f"  Avg E2E Delay: {e2e_delay_stats['mean']:.2f} seconds")
    else:
        e2e_delay_stats_results[speed_sim] = None
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

    throughput_results[speed_sim] = throughput_bps
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

    avg_reward_results[speed_sim] = avg_reward_by_direction
    avg_success_rate_results[speed_sim] = avg_success_rate_by_direction
    print(f"  Avg Reward by Direction: {avg_reward_by_direction}")
    print(f"  Avg Success Rate by Direction: {avg_success_rate_by_direction}")


    # Bloom Filter 'On' Bits
    on_bits_counts = []
    if Node.delivered_packets:
        for packet_info in Node.delivered_packets:
            # Retrieve the corresponding packet object to access the Bloom filter
            packet_obj = None
            destination_node = next((n for n in config.nodes if n.ip_address == destination.ip_address), None)
            if destination_node:
                for p in destination_node.opp_dest_packet:
                    if p.creation_timestamp == packet_info.get('creation_timestamp'):
                        packet_obj = p
                        break

            if packet_obj and hasattr(packet_obj, 'visited_nodes') and isinstance(packet_obj.visited_nodes, BloomFilter):
                bloom_filter = packet_obj.visited_nodes
                on_bits_count = sum(bin(byte).count('1') for byte in bloom_filter.bit_array)
                on_bits_counts.append(on_bits_count)

    if on_bits_counts:
        avg_on_bits_stats = pd.Series(on_bits_counts).describe()
        avg_on_bits_stats_results[speed_sim] = avg_on_bits_stats
        print(f"  Avg Bloom Filter On Bits: {avg_on_bits_stats['mean']:.2f}")
    else:
        avg_on_bits_stats_results[speed_sim] = None
        print("  Avg Bloom Filter On Bits: N/A (no delivered packets with Bloom filter data)")


    # Minimum Remaining Energy
    energies = [node.energy for node in config.nodes]
    min_energy = min(energies) if energies else 0
    min_energy_results[speed_sim] = min_energy
    print(f"  Minimum Remaining Energy: {min_energy:.2f}%")


    # Maximum Hops Used
    max_hops = 0
    if Node.delivered_packets:
         hops_used = [p['hops_used'] for p in Node.delivered_packets if 'hops_used' in p]
         max_hops = max(hops_used) if hops_used else 0
    max_hops_results[speed_sim] = max_hops
    print(f"  Maximum Hops Used: {max_hops}")


    # Average Initial and Final TTL
    avg_initial_ttl = 0
    avg_final_ttl = 0
    if Node.delivered_packets:
        initial_ttls = [p['initial_ttl'] for p in Node.delivered_packets if 'initial_ttl' in p]
        final_ttls = [p['final_ttl'] for p in Node.delivered_packets if 'final_ttl' in p]
        avg_initial_ttl = np.mean(initial_ttls) if initial_ttls else 0
        avg_final_ttl = np.mean(final_ttls) if final_ttls else 0
    avg_initial_ttl_results[speed_sim] = avg_initial_ttl
    avg_final_ttl_results[speed_sim] = avg_final_ttl
    print(f"  Avg Initial TTL: {avg_initial_ttl:.2f}")
    print(f"  Avg Final TTL: {avg_final_ttl:.2f}")

print("\n--- Simulation Runs Complete ---")

# Initialize a dictionary to hold the summarized metrics
performance_metrics_summary = {}

# Iterate through each speed that was simulated
for speed in speeds_to_simulate:
    metrics_for_speed = {}

    # 1. PDR
    metrics_for_speed['PDR (%)'] = pdr_results.get(speed)

    # 2. End-to-End Delay Statistics
    e2e_delay_stats = e2e_delay_stats_results.get(speed)
    if e2e_delay_stats is not None:
        metrics_for_speed['E2E Delay Mean (s)'] = e2e_delay_stats['mean']
        metrics_for_speed['E2E Delay Median (s)'] = e2e_delay_stats['50%']
        metrics_for_speed['E2E Delay Std (s)'] = e2e_delay_stats['std']
        metrics_for_speed['E2E Delay Min (s)'] = e2e_delay_stats['min']
        metrics_for_speed['E2E Delay Max (s)'] = e2e_delay_stats['max']
    else:
        metrics_for_speed['E2E Delay Mean (s)'] = None
        metrics_for_speed['E2E Delay Median (s)'] = None
        metrics_for_speed['E2E Delay Std (s)'] = None
        metrics_for_speed['E2E Delay Min (s)'] = None
        metrics_for_speed['E2E Delay Max (s)'] = None


    # 3. Estimated Throughput
    metrics_for_speed['Estimated Throughput (bps)'] = throughput_results.get(speed)

    # 4. Direction-Based Rewards (store averages directly)
    avg_rewards = avg_reward_results.get(speed, {})
    metrics_for_speed['Avg Reward Closer (+1)'] = avg_rewards.get(1, None)
    metrics_for_speed['Avg Reward Stationary (0)'] = avg_rewards.get(0, None)
    metrics_for_speed['Avg Reward Away (-1)'] = avg_rewards.get(-1, None)

    # 5. Direction-Based Success Rates (store averages directly)
    avg_success_rates = avg_success_rate_results.get(speed, {})
    metrics_for_speed['Avg Success Rate Closer (+1)'] = avg_success_rates.get(1, None)
    metrics_for_speed['Avg Success Rate Stationary (0)'] = avg_success_rates.get(0, None)
    metrics_for_speed['Avg Success Rate Away (-1)'] = avg_success_rates.get(-1, None)

    # 6. Bloom Filter 'On' Bits Statistics
    on_bits_stats = avg_on_bits_stats_results.get(speed)
    if on_bits_stats is not None:
        metrics_for_speed['BF On Bits Mean'] = on_bits_stats['mean']
        metrics_for_speed['BF On Bits Median'] = on_bits_stats['50%']
        metrics_for_speed['BF On Bits Std'] = on_bits_stats['std']
        metrics_for_speed['BF On Bits Min'] = on_bits_stats['min']
        metrics_for_speed['BF On Bits Max'] = on_bits_stats['max']
    else:
        metrics_for_speed['BF On Bits Mean'] = None
        metrics_for_speed['BF On Bits Median'] = None
        metrics_for_speed['BF On Bits Std'] = None
        metrics_for_speed['BF On Bits Min'] = None
        metrics_for_speed['BF On Bits Max'] = None


    # 7. Minimum Remaining Energy
    metrics_for_speed['Min Remaining Energy (%)'] = min_energy_results.get(speed)

    # 8. Maximum Hops Used
    metrics_for_speed['Max Hops Used'] = max_hops_results.get(speed)

    # 9. Average Initial and Final TTL
    metrics_for_speed['Avg Initial TTL'] = avg_initial_ttl_results.get(speed)
    metrics_for_speed['Avg Final TTL'] = avg_final_ttl_results.get(speed)

    # 10. Simulation Duration
    metrics_for_speed['Simulation Duration (s)'] = simulation_duration_results.get(speed)

    # Store the collected metrics for this speed
    performance_metrics_summary[speed] = metrics_for_speed

# Convert the summary dictionary to a Pandas DataFrame for better visualization and analysis
performance_summary_df = pd.DataFrame.from_dict(performance_metrics_summary, orient='index')

# Display the summary DataFrame
print("\n=== Performance Metrics Summary Across Speeds ===")
print(performance_summary_df)  # Use print instead of display for compatibility


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

# Exclude Bloom Filter and Visited Nodes Size metrics as per instructions
bar_plot_bf_metrics = [
    'BF On Bits Mean'
]

# Generate Line Plots
print("Generating Line Plots for Performance Metrics vs. Node Speed:")
for metric in line_plot_metrics:
    if metric in performance_summary_df.columns:
        # Replace None with np.nan for plotting
        y = performance_summary_df[metric].replace({None: np.nan})
        plt.figure(figsize=(10, 6))
        plt.plot(performance_summary_df.index, y, marker='o', linestyle='-')
        plt.xlabel("Node Speed (m/s)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Node Speed")
        plt.grid(True)
        plt.xticks(performance_summary_df.index) # Ensure all speeds are shown as ticks
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metric '{metric}' not found in performance_summary_df.")

# Generate Bar Plots for Direction-Based Rewards
print("\nGenerating Bar Plots for Average Reward by Direction vs. Node Speed:")
plt.figure(figsize=(12, 8))
bar_width = 0.8 / len(bar_plot_direction_metrics_reward) # Dynamic bar width based on number of categories
x = np.arange(len(performance_summary_df.index))

for i, metric in enumerate(bar_plot_direction_metrics_reward):
     if metric in performance_summary_df.columns:
        # Replace None with np.nan for plotting
        y = performance_summary_df[metric].replace({None: np.nan})
        plt.bar(x + i * bar_width, y.fillna(0), bar_width, label=metric.replace('Avg Reward ', ''))

plt.xlabel("Node Speed (m/s)")
plt.ylabel("Average Reward")
plt.title("Average Reward by Direction vs. Node Speed")
plt.xticks(x + bar_width * (len(bar_plot_direction_metrics_reward) - 1) / 2, performance_summary_df.index, rotation=0) # Rotate x-axis labels
plt.legend(title="Direction")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()


# Generate Bar Plots for Direction-Based Success Rates
print("\nGenerating Bar Plots for Average Success Rate by Direction vs. Node Speed:")
plt.figure(figsize=(12, 8))
x = np.arange(len(performance_summary_df.index))
bar_width = 0.8 / len(bar_plot_direction_metrics_success)

for i, metric in enumerate(bar_plot_direction_metrics_success):
     if metric in performance_summary_df.columns:
        # Replace None with np.nan for plotting
        y = performance_summary_df[metric].replace({None: np.nan})
        plt.bar(x + i * bar_width, y.fillna(0), bar_width, label=metric.replace('Avg Success Rate ', ''))

plt.xlabel("Node Speed (m/s)")
plt.ylabel("Average Success Rate")
plt.title("Average Success Rate by Direction vs. Node Speed")
plt.xticks(x + bar_width * (len(bar_plot_direction_metrics_success) - 1) / 2, performance_summary_df.index, rotation=0) # Rotate x-axis labels
plt.legend(title="Direction")
plt.ylim(0, 1.1) # Success rate is between 0 and 1, add a little buffer
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Generate Bar Plots for Bloom Filter Stats
print("\nGenerating Bar Plots for Bloom Filter On Bits Mean vs. Node Speed:")
plt.figure(figsize=(12, 8))
x = np.arange(len(performance_summary_df.index))
bar_width = 0.8 / len(bar_plot_bf_metrics)


for i, metric in enumerate(bar_plot_bf_metrics):
     if metric in performance_summary_df.columns:
        # Replace None with np.nan for plotting
        y = performance_summary_df[metric].replace({None: np.nan})
        plt.bar(x + i * bar_width, y.fillna(0), bar_width, label=metric.replace('Mean', ''))

plt.xlabel("Node Speed (m/s)")
plt.ylabel("Number of On bits")
plt.title("Bloom Filter vs. Node Speed")
plt.xticks(x + bar_width * (len(bar_plot_bf_metrics) - 1) / 2, performance_summary_df.index, rotation=0) # Rotate x-axis labels
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
