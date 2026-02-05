# ============================================================
# MANET TEMPORAL-CONSISTENCY ROUTING (NO MAB)
# TGNN-based Lifetime Prediction + TempReasoner Logic
# ============================================================
# This implementation is BASED ON YOUR SHARED SIMULATION CODE
# Changes:
#   1. ❌ Removed MAB / UCB completely
#   2. ✅ Forwarder selection = Temporal Consistency Optimization
#   3. ✅ TGNN used ONLY for link lifetime prediction
#   4. ✅ Path-aware, loop-averse, future-aware routing
# ============================================================

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import deque
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
NEIGHBOR_RADIUS = 250  # Changed from 350 to 250

# ================= TGNN: Link Lifetime Predictor =================
class TGNNLinkLifetime(nn.Module):
    def __init__(self, feature_dim=9, hidden_dim=64, window=3):  # window reduced from 5 to 3
        super().__init__()
        self.gcn1 = GCNConv(feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.window = window

    def forward(self, x_seq, edge_index_seq):
        outs = []
        for t in range(self.window):
            x_t = x_seq[t]
            edge_t = edge_index_seq[t]
            # Fix: If only one node, use self-loop edge index
            if x_t.size(0) == 1:
                edge_t = torch.tensor([[0],[0]], dtype=torch.long, device=x_t.device)
            h = F.relu(self.gcn1(x_t, edge_t))
            h = F.relu(self.gcn2(h, edge_t))
            outs.append(h.unsqueeze(0))
        hseq = torch.cat(outs, dim=0).permute(1, 0, 2)
        out, _ = self.gru(hseq)
        return self.fc(out[:, -1, :]).squeeze(-1)

# ================= NODE =================
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.x = random.uniform(0, 1000)
        self.y = random.uniform(0, 1000)
        self.z = random.uniform(0, 1000)
        self.energy = 100.0
        self.queue = []
        self.pos_history = deque(maxlen=5)
        self.link_hist = {}
        self.tgnn_hist = {}
        self.tgnn = TGNNLinkLifetime().to(DEVICE)
        self.tgnn_optimizer = torch.optim.Adam(self.tgnn.parameters(), lr=1e-3)
        self.tgnn_loss_fn = nn.MSELoss()
        self.last_pred = {}  # nid -> (pred, t)
        self.last_true = {}  # nid -> (true, t)
        self.tgnn_losses = []  # Track TGNN loss for plotting
        # Assign unique IP and MAC addresses
        self.ip = f"10.0.0.{node_id+1}"
        self.mac = f"00:00:00:00:00:{node_id:02x}"

    def update_position(self, t):
        self.pos_history.append((t, self.x, self.y, self.z))

    def update_link(self, neighbor, t):
        dx = neighbor.x - self.x
        dy = neighbor.y - self.y
        dz = neighbor.z - self.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if neighbor.node_id not in self.link_hist:
            self.link_hist[neighbor.node_id] = deque(maxlen=3)  # window reduced to 3
        self.link_hist[neighbor.node_id].append((t, dist))

        feat = np.array([
            dx, dy, dz,
            0.0, 0.0, 0.0,
            neighbor.energy,
            t - self.link_hist[neighbor.node_id][0][0],
            self.distance_trend(neighbor.node_id)
        ], dtype=np.float32)

        self.tgnn_hist.setdefault(neighbor.node_id, deque(maxlen=3)).append(feat)  # window reduced to 3
        # Online learning: if enough history, predict and store pred/true for update
        if len(self.tgnn_hist[neighbor.node_id]) == 3:
            x_seq = torch.stack([torch.tensor(f, device=DEVICE).unsqueeze(0) for f in self.tgnn_hist[neighbor.node_id]])
            # Fix: Always use valid edge indices for each time step
            edge_list = []
            for _ in range(3):
                if x_seq[0].size(0) == 1:
                    edge = torch.tensor([[0],[0]], dtype=torch.long, device=DEVICE)
                else:
                    edge = torch.tensor([[0,1],[1,0]], dtype=torch.long, device=DEVICE)
                edge_list.append(edge)
            with torch.no_grad():
                pred = float(self.tgnn(x_seq, edge_list)[0])
            self.last_pred[neighbor.node_id] = (pred, t)
            # If previous prediction exists and link broke (distance > NEIGHBOR_RADIUS), update
            if neighbor.node_id in self.link_hist and len(self.link_hist[neighbor.node_id]) >= 2:
                prev_t, prev_dist = self.link_hist[neighbor.node_id][-2]
                curr_t, curr_dist = self.link_hist[neighbor.node_id][-1]
                # If link just broke (distance > NEIGHBOR_RADIUS), treat as ground truth
                if prev_dist <= NEIGHBOR_RADIUS and curr_dist > NEIGHBOR_RADIUS:
                    true_lifetime = curr_t - prev_t
                    self.last_true[neighbor.node_id] = (true_lifetime, curr_t)
                    pred_val = torch.tensor([pred], device=DEVICE)
                    true_val = torch.tensor([true_lifetime], device=DEVICE)
                    loss = self.tgnn_loss_fn(pred_val, true_val)
                    self.tgnn_optimizer.zero_grad()
                    loss.backward()
                    self.tgnn_optimizer.step()
                    self.tgnn_losses.append(loss.item())
                    print(f"TGNN Online Loss (Node {self.node_id}, Link {neighbor.node_id}): {loss.item():.4f}")
        # Remove: self.last_feats = best_feats
        # Remove: return best

    def distance_trend(self, nid):
        h = self.link_hist.get(nid, None)
        if h is None or len(h) < 2:
            return 0.0
        (t1, d1), (t0, d0) = h[-1], h[-2]
        return (d1 - d0) / max(t1 - t0, 1e-3)

    def predict_lifetime(self, nid):
        hist = self.tgnn_hist.get(nid, None)
        if hist and len(hist) == 3:
            x_seq = torch.stack([torch.tensor(f).unsqueeze(0) for f in hist])
            # Fix: Always use valid edge indices for each time step
            edge_list = []
            for _ in range(3):
                if x_seq[0].size(0) == 1:
                    edge = torch.tensor([[0],[0]], dtype=torch.long)
                else:
                    edge = torch.tensor([[0,1],[1,0]], dtype=torch.long)
                edge_list.append(edge)
            with torch.no_grad():
                return float(self.tgnn(x_seq, edge_list)[0])
        return 0.0

# ================= TEMPORAL CONSISTENCY ROUTER (with real-world checks) =================
class TemporalRouter:
    def __init__(self, nodes, radius=NEIGHBOR_RADIUS, obstacles=None, epsilon=0.1):
        self.nodes = nodes
        self.radius = radius
        self.obstacles = obstacles if obstacles else []
        # Learnable weights for temporal consistency
        self.tc_weights = nn.Parameter(torch.tensor([2.0, 1.5, 1.0, 0.5], dtype=torch.float32, device=DEVICE))
        self.tc_optimizer = torch.optim.Adam([self.tc_weights], lr=1e-2)
        self.tc_loss_fn = lambda reward, score: -reward * score  # Policy gradient style
        self.tc_losses = []  # Track TempReasoner loss for plotting
        self.epsilon = epsilon  # ϵ-greedy exploration parameter

    def neighbors(self, node):
        nbrs = []
        for n in self.nodes:
            if n.node_id == node.node_id:
                continue
            d = np.linalg.norm([n.x-node.x, n.y-node.y, n.z-node.z])
            if d <= self.radius and is_link_feasible(node, n, self.obstacles):
                nbrs.append(n)
        return nbrs

    def select_forwarder(self, src, dst):
        src_pos = np.array([src.x, src.y, src.z])
        dst_pos = np.array([dst.x, dst.y, dst.z])
        src_dist = np.linalg.norm(src_pos - dst_pos)

        nbrs = self.neighbors(src)
        # Filter neighbors that make progress and pass MAC contention
        valid_nbrs = []
        feats_list = []
        scores = []
        for n in nbrs:
            n_pos = np.array([n.x, n.y, n.z])
            n_dist = np.linalg.norm(n_pos - dst_pos)
            if n_dist >= src_dist:
                continue
            if not mac_contention(src, nbrs):
                continue
            progress = (src_dist - n_dist) / (src_dist + 1e-6)
            lifetime = src.predict_lifetime(n.node_id)
            stability = 1.0 / (1.0 + abs(src.distance_trend(n.node_id)))
            energy_term = n.energy / 100.0
            feats = torch.tensor([progress, lifetime, stability, energy_term], dtype=torch.float32, device=DEVICE)
            score = torch.dot(self.tc_weights, feats).item()
            valid_nbrs.append(n)
            feats_list.append(feats)
            scores.append(score)

        if not valid_nbrs:
            self.last_feats = None
            return None

        # ϵ-greedy: with probability epsilon, explore (random neighbor); else exploit (best score)
        if random.random() < self.epsilon:
            idx = random.randrange(len(valid_nbrs))
            best = valid_nbrs[idx]
            best_feats = feats_list[idx]
        else:
            idx = int(np.argmax(scores))
            best = valid_nbrs[idx]
            best_feats = feats_list[idx]

        self.last_feats = best_feats
        return best

    def online_update(self, reward):
        # reward: +1 for delivery, -1 for failure, or function of hops
        if self.last_feats is not None:
            score = torch.dot(self.tc_weights, self.last_feats)
            loss = self.tc_loss_fn(reward, score)
            self.tc_optimizer.zero_grad()
            loss.backward()
            self.tc_optimizer.step()
            self.tc_losses.append(loss.item())
            print(f"TempReasoner Online Loss: {loss.item():.4f}")

# ================= OBSTACLE MODEL =================
class Obstacle:
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    def blocks(self, x1, y1, z1, x2, y2, z2):
        # Simple sphere intersection check
        # Returns True if the line between (x1,y1,z1) and (x2,y2,z2) passes through the obstacle
        # For realism, use more advanced geometry in production
        cx, cy, cz = self.x, self.y, self.z
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        fx, fy, fz = x1 - cx, y1 - cy, z1 - cz
        a = dx*dx + dy*dy + dz*dz
        b = 2 * (fx*dx + fy*dy + fz*dz)
        c = fx*fx + fy*fy + fz*fz - self.radius*self.radius
        discriminant = b*b - 4*a*c
        return discriminant >= 0

# ================= RADIO PROPAGATION MODEL =================
def path_loss(d, freq=2.4e9):
    # Free-space path loss (FSPL) in dB
    if d < 1.0:
        d = 1.0
    c = 3e8
    fspl = 20 * math.log10(d) + 20 * math.log10(freq) - 147.55
    return fspl

def is_link_feasible(n1, n2, obstacles, tx_power_dbm=0, rx_sensitivity_dbm=-90):
    # Check if link is blocked by obstacles
    for obs in obstacles:
        if obs.blocks(n1.x, n1.y, n1.z, n2.x, n2.y, n2.z):
            return False
    # Check if received power is above sensitivity
    d = np.linalg.norm([n1.x-n2.x, n1.y-n2.y, n1.z-n2.z])
    # Only consider links within NEIGHBOR_RADIUS
    if d > NEIGHBOR_RADIUS:
        return False
    loss = path_loss(d)
    rx_power = tx_power_dbm - loss
    return rx_power >= rx_sensitivity_dbm

# ================= MAC CONTENTION MODEL =================
def mac_contention(n, neighbors):
    # Simulate MAC contention: random backoff, possible collision
    # For realism, use CSMA/CA or TDMA in production
    if len(neighbors) == 0:
        return True
    # 10% chance of collision if >3 neighbors
    if len(neighbors) > 3 and random.random() < 0.1:
        return False
    return True

# ================= ENERGY MODEL =================
def realistic_energy_consumption(n, d, tx=True):
    # More realistic: energy depends on distance and radio state
    # Assume 0.05mJ per meter for TX, 0.02mJ for RX
    if tx:
        n.energy -= 0.05 * d / 1000.0
    else:
        n.energy -= 0.02 * d / 1000.0

# ================= MOBILITY MODEL (with obstacles) =================
class SteadyStateRandomWaypoint:
    def __init__(self, nodes, speed=30, bound=1000, obstacles=None):
        self.nodes = nodes
        self.speed = speed
        self.bound = bound
        self.obstacles = obstacles if obstacles else []
        self.targets = {n.node_id: self._rand_point() for n in nodes}

    def _rand_point(self):
        # Avoid obstacles when picking a target
        while True:
            x, y, z = (
                random.uniform(0, self.bound),
                random.uniform(0, self.bound),
                random.uniform(0, self.bound)
            )
            blocked = any(
                math.sqrt((x-o.x)**2 + (y-o.y)**2 + (z-o.z)**2) < o.radius
                for o in self.obstacles
            )
            if not blocked:
                return (x, y, z)

    def step(self, dt=1.0):
        for n in self.nodes:
            tx, ty, tz = self.targets[n.node_id]
            dx, dy, dz = tx - n.x, ty - n.y, tz - n.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-6:
                self.targets[n.node_id] = self._rand_point()
                continue
            move = min(self.speed * dt, dist)
            # Avoid obstacles: if next step is inside obstacle, pick new target
            nx, ny, nz = n.x + (dx/dist)*move, n.y + (dy/dist)*move, n.z + (dz/dist)*move
            blocked = any(
                math.sqrt((nx-o.x)**2 + (ny-o.y)**2 + (nz-o.z)**2) < o.radius
                for o in self.obstacles
            )
            if blocked:
                self.targets[n.node_id] = self._rand_point()
                continue
            n.x, n.y, n.z = nx, ny, nz

# ================= NOISE MODEL =================
def apply_noise(delivered_packets, noise_level):
    if noise_level <= 0:
        return delivered_packets
    keep = int((1 - noise_level) * len(delivered_packets))
    if keep <= 0:
        return []
    return random.sample(delivered_packets, keep)

# ================= SIMULATION =================
class Packet:
    def __init__(self, src_ip, dst_ip, src_mac, dst_mac, src_port, dst_port, ttl, payload):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_mac = src_mac
        self.dst_mac = dst_mac
        self.src_port = src_port
        self.dst_port = dst_port
        self.ttl = ttl
        self.payload = payload
        self.hops = 0

def simulate(nodes=80, steps=40, speed=50, noise_level=0.2, epsilon=0.1):
    # Increased steps to 40, speed to 50 for more link break events
    obstacles = [
        Obstacle(500, 500, 500, 100),
        Obstacle(200, 800, 400, 80)
    ]
    nodes = [Node(i) for i in range(nodes)]
    router = TemporalRouter(nodes, obstacles=obstacles, epsilon=epsilon)
    mobility = SteadyStateRandomWaypoint(nodes, speed=speed, obstacles=obstacles)

    delivered_packets = []
    delivered_info = []  # (pkt, delivery_time, gen_time, hops)
    pkt_gen_time = {}    # (src_ip, dst_ip, payload) -> (gen_time, hops)

    for t in range(steps):
        mobility.step(dt=1.0)
        for n in nodes:
            n.update_position(t)

        # --- TGNN: Update link state for all node pairs within NEIGHBOR_RADIUS ---
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                d = np.linalg.norm([n1.x-n2.x, n1.y-n2.y, n1.z-n2.z])
                if d <= NEIGHBOR_RADIUS:
                    n1.update_link(n2, t)
                    n2.update_link(n1, t)

        # Generate packets: one per node per step
        for src in nodes:
            dst = random.choice([n for n in nodes if n != src])
            pkt = Packet(
                src_ip=src.ip,
                dst_ip=dst.ip,
                src_mac=src.mac,
                dst_mac=dst.mac,
                src_port=12345,
                dst_port=54321,
                ttl=10,
                payload=f"Data from {src.node_id} to {dst.node_id}"
            )
            src.queue.append(pkt)
            pkt_gen_time[(pkt.src_ip, pkt.dst_ip, pkt.payload)] = (t, 0)

        for n in nodes:
            new_q = []
            for pkt in n.queue:
                # Drop if TTL expired
                if pkt.ttl <= 0:
                    continue
                # Check if delivered
                if n.ip == pkt.dst_ip and n.mac == pkt.dst_mac:
                    delivered_packets.append(pkt)
                    # End-to-end delay calculation
                    key = (pkt.src_ip, pkt.dst_ip, pkt.payload)
                    gen_time, _ = pkt_gen_time.get(key, (t, 0))
                    delivered_info.append((pkt, t, gen_time, pkt.hops))
                    # Online update: reward +1 for delivery, penalize by hops
                    reward = 1.0 - 0.05 * pkt.hops
                    router.online_update(reward)
                    continue
                # Find destination node
                dst = next((node for node in nodes if node.ip == pkt.dst_ip and node.mac == pkt.dst_mac), None)
                if not dst:
                    continue
                nxt = router.select_forwarder(n, dst)
                if nxt:
                    # Realistic energy consumption
                    d_dist = np.linalg.norm([n.x-nxt.x, n.y-nxt.y, n.z-nxt.z])
                    realistic_energy_consumption(n, d_dist, tx=True)
                    realistic_energy_consumption(nxt, d_dist, tx=False)
                    # Prepare next hop packet
                    fwd_pkt = Packet(
                        src_ip=pkt.src_ip,
                        dst_ip=pkt.dst_ip,
                        src_mac=nxt.mac,
                        dst_mac=pkt.dst_mac,
                        src_port=pkt.src_port,
                        dst_port=pkt.dst_port,
                        ttl=pkt.ttl - 1,
                        payload=pkt.payload
                    )
                    fwd_pkt.hops = pkt.hops + 1
                    nxt.queue.append(fwd_pkt)
                    # Update hops for delay tracking
                    key = (pkt.src_ip, pkt.dst_ip, pkt.payload)
                    pkt_gen_time[key] = (pkt_gen_time.get(key, (t, 0))[0], fwd_pkt.hops)
                else:
                    # Not forwarded, keep in queue if TTL not expired
                    pkt.ttl -= 1
                    if pkt.ttl > 0:
                        pkt.hops += 1
                        new_q.append(pkt)
                    # Online update: reward -1 for failure
                    router.online_update(-1.0)
            n.queue = new_q

    delivered_noisy = apply_noise(delivered_packets, noise_level)
    pdr = len(delivered_noisy) / max(1, len(delivered_packets))
    print(f"Delivered: {len(delivered_noisy)} / {len(delivered_packets)} | PDR={pdr:.3f}")
    print(f"True delivered (pre-noise): {len(delivered_packets)} / {steps * len(nodes)} | True PDR={len(delivered_packets)/(steps*len(nodes)):.3f}")

    # --- Throughput and End-to-End Delay Calculation ---
    # Throughput: total delivered packets / total simulation time (steps)
    throughput = len(delivered_packets) / steps
    print(f"Throughput: {throughput:.3f} packets/step")

    # End-to-end delay: average (delivery_time - gen_time) over delivered packets
    if delivered_info:
        delays = [delivery_time - gen_time for (_, delivery_time, gen_time, _) in delivered_info]
        avg_delay = sum(delays) / len(delays)
        print(f"Average End-to-End Delay: {avg_delay:.3f} steps")
    else:
        avg_delay = 0.0
        print("Average End-to-End Delay: N/A")

    # --- Plot Losses ---
    # Gather TGNN and TempReasoner losses
    tgnn_losses = []
    for n in nodes:
        tgnn_losses.extend(n.tgnn_losses)
    tc_losses = router.tc_losses

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    if tgnn_losses:
        plt.plot(tgnn_losses, label="TGNN Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.title("TGNN Online Loss")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No TGNN Losses", ha='center', va='center')
    plt.subplot(1,2,2)
    if tc_losses:
        plt.plot(tc_losses, label="TempReasoner Loss", color='orange')
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.title("TempReasoner Online Loss")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No TempReasoner Losses", ha='center', va='center')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate()
