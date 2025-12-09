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

class GNNModel(nn.Module):
    """Very deep GCN-based GNN for neighbor scoring with more layers and minimal dropout."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(8)])
        self.out_layer = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = F.leaky_relu(conv(x, edge_index), negative_slope=0.1)
            x = self.dropout(x)
        out = self.out_layer(x, edge_index)
        return out.squeeze(-1)  # [N] scores

class OnlineGNN:
    """Online GNN agent for neighbor discovery and scoring using real GNN."""
    def __init__(self, node, all_nodes, input_dim=5, hidden_dim=1024, neighbor_radius=700):
        self.node = node
        self.all_nodes = all_nodes
        self.neighbor_radius = neighbor_radius
        self.neighbors = set()
        self.neighbor_features = {}  # node_id: feature vector
        self.gnn = GNNModel(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01)
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
        x = torch.tensor(x, dtype=torch.float32)
        # Fully connect self to neighbors (undirected)
        edge_index = []
        for nid in self.neighbor_features:
            i = 0  # self
            j = id_to_idx[nid]
            edge_index.append([i, j])
            edge_index.append([j, i])
        if not edge_index:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
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

    def predict_best_forwarder(self):
        embeddings = self.compute_embeddings()
        if not embeddings:
            return None
        best_nid = max(embeddings, key=lambda k: embeddings[k])
        return best_nid, embeddings[best_nid]

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
        targets = torch.zeros(x.size(0))
        for nid, val in feedback.items():
            idx = id_to_idx.get(nid, None)
            if idx is not None and idx < len(targets):
                targets[idx] = val
        pred = self.gnn(x, edge_index)
        loss = F.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def mab_select_forwarder(self, candidates):
        """Select best forwarder using UCB with very high exploration bonus and reward shaping."""
        self.total_mab_selections += 1
        ucb_scores = {}
        for nid in candidates:
            count = self.mab_counts.get(nid, 0)
            reward = self.mab_rewards.get(nid, 0)
            if count == 0:
                ucb_scores[nid] = float('inf')
            else:
                avg_reward = reward / count
                # Even higher exploration bonus (factor 32)
                ucb_scores[nid] = avg_reward + math.sqrt(32 * math.log(self.total_mab_selections + 1) / count)
        # Select the neighbor with the highest UCB score
        return max(ucb_scores, key=ucb_scores.get)

    def mab_update(self, neighbor_id, reward, src_node=None, dst_node=None, forwarder_node=None):
        # Reward shaping: much higher bonus for progress
        shaped_reward = reward
        if src_node is not None and dst_node is not None and forwarder_node is not None:
            src_pos = np.array([src_node.x, src_node.y, src_node.z])
            dst_pos = np.array([dst_node.x, dst_node.y, dst_node.z])
            fwd_pos = np.array([forwarder_node.x, forwarder_node.y, forwarder_node.z])
            src_dist = np.linalg.norm(src_pos - dst_pos)
            fwd_dist = np.linalg.norm(fwd_pos - dst_pos)
            if fwd_dist < src_dist:
                shaped_reward += 4.0  # Much higher partial reward for progress
        self.mab_counts[neighbor_id] = self.mab_counts.get(neighbor_id, 0) + 1
        self.mab_rewards[neighbor_id] = self.mab_rewards.get(neighbor_id, 0) + shaped_reward

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

    def __repr__(self):
        return f"Node({self.node_id})"

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

    def construct_oppdata_frame(self, dst_id, hops=0):
        # Build an opportunistic data frame to send to dst_id
        oppdata = OppData(self.node_id, dst_id)
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
            print(f"Node {self.node_id} received OppData from {oppdata.src_id}: {oppdata.data:.3f} (hops={frame.get('hops', 0)})")
            return True
        return False

    def forward_oppdata(self, config_nodes, spatial_grid, delivered_packets, max_hops=20):
        """Process the forwarding queue for multi-hop delivery."""
        new_queue = []
        for frame in self.forwarding_queue:
            oppdata = frame["payload"]
            hops = frame.get("hops", 0)
            if hops >= max_hops:
                continue  # Drop if exceeded max hops
            if self.receive_oppdata_frame(frame):
                delivered_packets.append((oppdata.src_id, oppdata.dst_id))
                continue  # Delivered
            # Not delivered, forward to next best neighbor
            self.gnn.update_neighbors()
            neighbor_scores = self.gnn.compute_embeddings()
            if neighbor_scores:
                # Use all available neighbors as candidates
                candidates = list(neighbor_scores.keys())
                best_forwarder_id = self.gnn.mab_select_forwarder(candidates)
                if best_forwarder_id in self.gnn.neighbors:
                    forwarder = next((n for n in config_nodes if n.node_id == best_forwarder_id), None)
                    if forwarder:
                        # Create a new frame with incremented hop count
                        new_frame = self.construct_oppdata_frame(oppdata.dst_id, hops=hops+1)
                        new_frame["payload"] = oppdata  # Preserve original data
                        forwarder.forwarding_queue.append(new_frame)
                        reward = 1.0 if forwarder.node_id == oppdata.dst_id else 0.0
                        # Pass src, dst, forwarder for reward shaping
                        self.gnn.mab_update(best_forwarder_id, reward, src_node=self, dst_node=next((n for n in config_nodes if n.node_id == oppdata.dst_id), None), forwarder_node=forwarder)
                        # Optionally print forwarding
                        print(f"Node {self.node_id} forwarded OppData to {oppdata.dst_id} via {best_forwarder_id} (hops={hops+1})")
                else:
                    # No valid neighbor, drop
                    pass
            else:
                # No neighbors, drop
                pass
        self.forwarding_queue = new_queue  # Clear queue after processing

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
    def __init__(self, src_id, dst_id, data=None):
        self.src_id = src_id
        self.dst_id = dst_id
        self.data = data if data is not None else random.random()

    def __repr__(self):
        return f"OppData(src_id={self.src_id}, dst_id={self.dst_id}, data={self.data:.3f})"

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

if __name__ =="__main__":
    config = Configuration(num_nodes=120)  # Increased node count for higher density
    mobility = SteadyStateRandomWaypointMobility(config.nodes, speed=20)  # Lower speed for more stable links
    cell_size = 250
    spatial_grid = SpatialGrid(cell_size)
    steps = 5
    packets_per_step = 10  # Send only 10 packets per step
    pdr_list = []

    # --- Pre-training phase for GNN ---
    pretrain_steps = 100
    for _ in range(pretrain_steps):
        mobility.step()
        spatial_grid.assign_nodes(config.nodes)
        for node in config.nodes:
            node.gnn.update_neighbors()
            feedback = {}
            valid_neighbors = set(node.gnn.neighbor_features.keys())
            for neighbor_id in valid_neighbors:
                feedback[neighbor_id] = random.choice([0.0, 1.0])
            node.gnn.online_update(feedback)

    # --- Main simulation ---
    for step in range(steps):
        mobility.step()
        spatial_grid.assign_nodes(config.nodes)
        for sender in config.nodes:
            frame = sender.construct_frame()
            receivers = spatial_grid.get_receivers(sender, sender.gnn.neighbor_radius)
            for receiver in receivers:
                receiver.receive_frame(frame)
        for node in config.nodes:
            print(f"Node {node.node_id} neighbors: {len(node.gnn.get_neighbors())} -> {sorted(node.gnn.get_neighbors())}")
        print("-" * 40)
        #plot_node_positions(config.nodes, step)

        for node in config.nodes:
            node.gnn.update_neighbors()
            feedback = {}
            valid_neighbors = set(node.gnn.neighbor_features.keys())
            for neighbor_id in valid_neighbors:
                feedback[neighbor_id] = random.choice([0.0, 1.0])
            node.gnn.online_update(feedback)

        delivered_packets = []
        for _ in range(packets_per_step):
            src, dst = random.sample(config.nodes, 2)
            opp_frame = src.construct_oppdata_frame(dst.node_id, hops=0)
            src.forwarding_queue.append(opp_frame)
        for node in config.nodes:
            node.forward_oppdata(config.nodes, spatial_grid, delivered_packets, max_hops=50)  # Increased max_hops
        pdr = len(delivered_packets) / packets_per_step
        pdr_list.append(pdr)
        print(f"Step {step}: PDR = {pdr:.2f}")
        print("="*60)

    # Print overall PDR after all steps
    print(f"Average PDR over {steps} steps: {np.mean(pdr_list):.2f}")

    # Plot PDR over steps
    plt.figure(figsize=(7,4))
    plt.plot(range(steps), pdr_list, marker='o')
    plt.title("Packet Delivery Ratio (PDR) per Step")
    plt.xlabel("Step")
    plt.ylabel("PDR")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






