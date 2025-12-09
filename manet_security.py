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

class GNNModel(nn.Module):
    """GCN-based GNN for neighbor scoring."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        out = self.conv2(x, edge_index)
        return out.squeeze(-1)  # [N] scores

class OnlineGNN:
    """Online GNN agent for neighbor discovery and scoring using real GNN."""
    def __init__(self, node, all_nodes, input_dim=5, hidden_dim=16, neighbor_radius=250):
        self.node = node
        self.all_nodes = all_nodes
        self.neighbor_radius = neighbor_radius
        self.neighbors = set()
        self.neighbor_features = {}  # node_id: feature vector
        self.gnn = GNNModel(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01)
        self.last_edge_index = None
        self.last_x = None

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
        # Return scores for neighbors only (not self)
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
            if nid in id_to_idx:
                targets[id_to_idx[nid]] = val
        pred = self.gnn(x, edge_index)
        loss = F.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    config = Configuration(num_nodes=100)
    mobility = SteadyStateRandomWaypointMobility(config.nodes, speed=40)
    cell_size = 250
    spatial_grid = SpatialGrid(cell_size)
    for step in range(5):
        mobility.step()
        spatial_grid.assign_nodes(config.nodes)
        # Each node constructs a frame and "sends" only to nodes within radio range
        for sender in config.nodes:
            frame = sender.construct_frame()
            receivers = spatial_grid.get_receivers(sender, sender.gnn.neighbor_radius)
            for receiver in receivers:
                receiver.receive_frame(frame)
        # Print neighbor tables
        for node in config.nodes:
            print(f"Node {node.node_id} neighbors: {len(node.gnn.get_neighbors())} -> {sorted(node.gnn.get_neighbors())}")
        print("-" * 40)
        # Plot node positions with color difference and numbering
        plot_node_positions(config.nodes, step)

        # --- Feedback collection and online_update integration ---
        # Simulate feedback: for each node, assign random feedback to its neighbors
        for node in config.nodes:
            node.gnn.update_neighbors()  # Ensure neighbor_features are up-to-date
            feedback = {}
            for neighbor_id in node.gnn.neighbors:
                # Simulate feedback: 1 for "success", 0 for "fail" (random for demo)
                feedback[neighbor_id] = random.choice([0.0, 1.0])
            node.gnn.online_update(feedback)






