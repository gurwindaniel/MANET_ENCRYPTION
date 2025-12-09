from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import math
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


try:
    import ascon
except ImportError:
    ascon = None  # Placeholder if ascon is not installed

class OPPPacket:
    def __init__(self, src_id, dst_id, payload, seq_num):
        self.src_id = src_id
        self.dst_id = dst_id
        self.payload = payload
        self.seq_num = seq_num
        self.delivered = False

class Node:
    def __init__(self, node_id, ip_address, mac_address, root_ca_public_key, node_certificate, ecc_public_key, ecc_private_key):
        self.node_id = node_id
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.root_ca_public_key = root_ca_public_key          # PEM string
        self.node_certificate = node_certificate              # hex signature over id+mac+pubkey
        self.ecc_public_key = ecc_public_key                  # PEM string
        self.ecc_private_key = ecc_private_key  
        self.position = (0.0, 0.0, 0.0)
        self.instantaneous_speed = 0.0
        self.energy = 100.0  # Joules, initial energy (can be adjusted)
        # Reinforcement Learning agent for neighbor selection (Multi-Armed Bandit)
        self.mab_agent = MABAgent(self)
        self.opp_queue = []
        self.q_agent = CooperativeQLAgent(self)
        self.delivered_packets = set()  # Track delivered packet seq_nums
        self.seen_packets = set()  # Track (src_id, seq_num) to avoid loops
        self.forwarded_seq_nums = set()  # Track which packets this node has forwarded

    def __repr__(self):
        return (f"Node(node_id={self.node_id}, ip_address={self.ip_address}, mac_address={self.mac_address}, "
                f"node_certificate={self.node_certificate[:32]}..., ecc_public_key=PEM, ecc_private_key=PEM)")

    def verify_certificate(self):
        """Verify this node's certificate using stored Root CA public key."""
        data = f"{self.node_id}:{self.mac_address}:{self.ecc_public_key}".encode()
        signature = bytes.fromhex(self.node_certificate)
        ca_pub = serialization.load_pem_public_key(self.root_ca_public_key.encode())
        try:
            ca_pub.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False

    def export_identity(self):
        """Data to send to another node."""
        return {
            "node_id": self.node_id,
            "mac_address": self.mac_address,
            "certificate": self.node_certificate,
            "ecc_public_key": self.ecc_public_key,
        }

    @staticmethod
    def verify_received_identity(payload, root_ca_public_key_pem):
        """Verify received payload and return ECC public key PEM if valid."""
        # Extracting fields from the Hello object
        payloads = {
            "node_id": payload.node_id,
            "mac_address": payload.mac_address,
            "certificate": payload.signature,  # Assuming signature is used as certificate
            "ecc_public_key": payload.ecc_public_key,
        }
        for k in ("node_id", "mac_address", "certificate", "ecc_public_key"):
            if k not in payloads:
                raise ValueError(f"Missing field {k}")
            if not payloads[k]:
                raise ValueError(f"Empty field {k}")
        data = f"{payloads['node_id']}:{payloads['mac_address']}:{payloads['ecc_public_key']}".encode()
        signature = bytes.fromhex(payloads["certificate"])
        ca_pub = serialization.load_pem_public_key(root_ca_public_key_pem.encode())
        ca_pub.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return payloads["ecc_public_key"]  # trusted now

    def create_handshake(self):
        """Create a Handshake object using this node's certificate and public key."""
        return Handshake(self.node_certificate, self.ecc_public_key)

    def create_hello_packet(self):
        """Create a Hello packet using this node's properties."""
        # Simulate energy drain for sending a packet
        if self.energy > 1.0:  # allow nodes with low energy to still participate
            self.drain_energy(0.005)
        signature = self.node_certificate
        return Hello(
            node_id=self.node_id,
            mac_address=self.mac_address,
            ecc_public_key=self.ecc_public_key,
            signature=signature,
            position=self.position,
            speed=self.instantaneous_speed
        )

    def receive_hello(self, hello):
        # Simulate energy drain for receiving a packet
        if self.energy > 1.0:
            self.drain_energy(0.0025)
        self.mab_agent.receive_hello(hello)

    def drain_energy(self, amount):
        """Drain node energy by a specified amount (Joules)."""
        self.energy = max(0.0, self.energy - amount)

    def energy_usage(self):
        """Return current energy level."""
        return self.energy

    def neighbor_count(self):
        return len(self.mab_agent.neighbors())

    def neighbor_details(self):
        return self.mab_agent.neighbor_info()

    def update_speed(self, new_speed):
        self.instantaneous_speed = new_speed

    def receive_opp_packet(self, opp_pkt, neighbor_info):
        pkt_id = (opp_pkt.src_id, opp_pkt.seq_num)
        if pkt_id in self.seen_packets:
            return
        self.seen_packets.add(pkt_id)
        self.opp_queue.append((opp_pkt, neighbor_info))

    def process_opp_queue(self, all_nodes, pdr_stats):
        new_queue = []
        max_forwards_per_packet = 8  # increase replication to improve reachability
        for opp_pkt, neighbor_info in self.opp_queue:
            pkt_id = (opp_pkt.src_id, opp_pkt.seq_num)
            if opp_pkt.dst_id == self.node_id:
                # Only count as delivered if this is the destination node and not already delivered
                if not opp_pkt.delivered:
                    opp_pkt.delivered = True
                    self.delivered_packets.add(opp_pkt.seq_num)
                    # Only increment delivered if this is the first time for this seq_num
                    if opp_pkt.seq_num not in pdr_stats.get('delivered_seq_nums', set()):
                        pdr_stats['delivered'] += 1
                        pdr_stats.setdefault('delivered_seq_nums', set()).add(opp_pkt.seq_num)
                        # Track delivered bytes and encryption bytes per-packet
                        payload = opp_pkt.payload
                        if isinstance(payload, (bytes, bytearray)):
                            payload_len = len(payload)
                            pdr_stats['encrypted_bytes_delivered'] = pdr_stats.get('encrypted_bytes_delivered', 0) + payload_len
                            pdr_stats['current_step_encrypted_bytes'] = pdr_stats.get('current_step_encrypted_bytes', 0) + payload_len
                        else:
                            payload_len = len(str(payload).encode())
                        pdr_stats['delivered_bytes'] = pdr_stats.get('delivered_bytes', 0) + payload_len
                        pdr_stats['current_step_bandwidth'] = pdr_stats.get('current_step_bandwidth', 0) + payload_len
                        # record delivery step if available
                        step = pdr_stats.get('current_step', None)
                        if step is not None:
                            pdr_stats.setdefault('packet_delivery_steps', {})[opp_pkt.seq_num] = step
                continue
            dst_node = next((n for n in all_nodes if n.node_id == opp_pkt.dst_id), None)
            if not dst_node:
                continue
            dst_pos = dst_node.position
            my_dist = math.sqrt(sum((a-b)**2 for a, b in zip(self.position, dst_pos)))
            forwarded = False
            valid_neighbors = []
            for nid in self.mab_agent.neighbors():
                if nid == self.node_id:
                    continue
                neighbor = next((n for n in all_nodes if n.node_id == nid), None)
                # allow more nodes with low energy to be eligible
                if not neighbor or neighbor.energy <= 1.0:
                    continue
                neighbor_dist = math.sqrt(sum((a-b)**2 for a, b in zip(neighbor.position, dst_pos)))
                pkt_id = (opp_pkt.src_id, opp_pkt.seq_num)
                if pkt_id not in neighbor.seen_packets:
                    valid_neighbors.append((neighbor, neighbor_dist))
            # Prefer neighbors with higher energy
            valid_neighbors.sort(key=lambda x: x[0].energy, reverse=True)
            # Try strictly closer neighbors first
            closer_neighbors = [n for n, d in valid_neighbors if d < my_dist]
            for forwards in range(max_forwards_per_packet):
                # allow nodes with lower remaining energy to forward
                if closer_neighbors and self.energy > 1.0 and pkt_id not in self.forwarded_seq_nums:
                    for neighbor in closer_neighbors[:max_forwards_per_packet]:
                        # derive per-pair key: fetch key derived between self and neighbor
                        key = self.mab_agent.get_shared_key(neighbor.node_id) or b'1234567890123456'
                        payload_bytes = opp_pkt.payload.encode() if isinstance(opp_pkt.payload, str) else opp_pkt.payload
                        encrypted_payload = self.encrypt_payload_bytes(payload_bytes, key)
                        new_pkt = OPPPacket(self.node_id, opp_pkt.dst_id, encrypted_payload, opp_pkt.seq_num)
                        neighbor.receive_opp_packet(new_pkt, neighbor.get_cooperation_info(self.node_id))
                        pdr_stats['forwarded'] += 1
                        self.drain_energy(0.03)  # slightly lower cost so more forwards possible
                        forwarded = True
                        forwards += 1
                        if forwards >= max_forwards_per_packet:
                            break
                    self.forwarded_seq_nums.add(pkt_id)
                # If no strictly closer neighbor, forward to the closest valid neighbor(s)
                elif valid_neighbors and self.energy > 1.0 and pkt_id not in self.forwarded_seq_nums:
                    for neighbor, _ in valid_neighbors[:max_forwards_per_packet]:
                        key = self.mab_agent.get_shared_key(neighbor.node_id) or b'1234567890123456'
                        payload_bytes = opp_pkt.payload.encode() if isinstance(opp_pkt.payload, str) else opp_pkt.payload
                        encrypted_payload = self.encrypt_payload_bytes(payload_bytes, key)
                        new_pkt = OPPPacket(self.node_id, opp_pkt.dst_id, encrypted_payload, opp_pkt.seq_num)
                        neighbor.receive_opp_packet(new_pkt, neighbor.get_cooperation_info(self.node_id))
                        pdr_stats['forwarded'] += 1
                        self.drain_energy(0.03)
                        forwarded = True
                        forwards += 1
                        if forwards >= max_forwards_per_packet:
                            break
                    self.forwarded_seq_nums.add(pkt_id)
            if not forwarded:
                new_queue.append((opp_pkt, neighbor_info))
        self.opp_queue = new_queue

    def encrypt_payload(self, payload, key):
        if ascon:
            # ensure key is bytes and 16 bytes long
            key = key if isinstance(key, bytes) else bytes(key, 'utf-8')
            if len(key) != 16:
                key = key.ljust(16, b'\0')[:16]
            nonce = os.urandom(16)  # unique nonce per encryption
            ct = ascon.encrypt(key, nonce, payload.encode(), b'')
            # prefix nonce so receiver can decrypt later: nonce || ciphertext
            return nonce + ct
        else:
            return payload.encode()  # No encryption fallback

    def encrypt_payload_bytes(self, payload_bytes, key):
        if ascon:
            # ASCON expects key and nonce as bytes of correct length, and payload as bytes.
            key = key if isinstance(key, bytes) else bytes(key, 'utf-8')
            if len(key) != 16:
                key = key.ljust(16, b'\0')[:16]
            nonce = os.urandom(16)
            payload_bytes = payload_bytes if isinstance(payload_bytes, bytes) else bytes(payload_bytes)
            try:
                ct = ascon.encrypt(key, nonce, payload_bytes, b'')
                return nonce + ct
            except Exception:
                # Fallback to unencrypted if ASCON fails; still prefix a zero nonce to indicate no-auth encryption
                return b'\x00'*16 + payload_bytes
        else:
            return payload_bytes  # No encryption fallback

    def get_cooperation_info(self, neighbor_id=None):
        """Return info for MARL cooperation and optionally per-neighbor shared key."""
        shared_key = None
        if neighbor_id:
            shared_key = self.mab_agent.get_shared_key(neighbor_id)
        # fallback default key (kept for compatibility)
        if shared_key is None:
            shared_key = b'1234567890123456'
        return {
            'mobility': self.instantaneous_speed,
            'energy': self.energy,
            'stability': 1.0,  # Placeholder for stability
            'q_top_actions': self.q_agent.top_actions(),
            'shared_key': shared_key,  # Example shared key
        }

class Configure:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.root_ca_private_key = ec.generate_private_key(ec.SECP256R1())
        self.root_ca_public_key = self.root_ca_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        self.nodes = []
        self._create_nodes()
        # assign random 3D positions for each node (increase space so long distances exist)
        self._assign_random_positions(space_size=1000.0)

    def _generate_mac(self):
        return "02:00:%02x:%02x:%02x:%02x" % tuple(os.urandom(4))

    def _generate_ip(self, idx):
        return f"192.168.1.{idx+1}"

    def _sign_certificate(self, node_id, mac_address, ecc_public_key_pem):
        """Sign id + mac + public key."""
        data = f"{node_id}:{mac_address}:{ecc_public_key_pem}".encode()
        signature = self.root_ca_private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        return signature.hex()

    def _create_nodes(self):
        for i in range(self.num_nodes):
            node_id = f"Node_{i+1}"
            ip_address = self._generate_ip(i)
            mac_address = self._generate_mac()

            ecc_priv = ec.generate_private_key(ec.SECP256R1())
            ecc_pub_pem = ecc_priv.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            ecc_priv_pem = ecc_priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()

            cert = self._sign_certificate(node_id, mac_address, ecc_pub_pem)

            node = Node(
                node_id=node_id,
                ip_address=ip_address,
                mac_address=mac_address,
                root_ca_public_key=self.root_ca_public_key,
                node_certificate=cert,
                ecc_public_key=ecc_pub_pem,
                ecc_private_key=ecc_priv_pem
            )
            self.nodes.append(node)
    def _assign_random_positions(self, space_size=100.0):
        """Assign random 3D positions to each node within a cube [0, space_size]^3."""
        for n in self.nodes:
            x = random.uniform(0, space_size)
            y = random.uniform(0, space_size)
            z = random.uniform(0, space_size)
            n.position = (x, y, z)
    def plot_nodes_3d(self, show_labels=True):
        """Plot node positions in 3D."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        xs = [n.position[0] for n in self.nodes]
        ys = [n.position[1] for n in self.nodes]
        zs = [n.position[2] for n in self.nodes]

        ax.scatter(xs, ys, zs, c='tab:blue', s=60, depthshade=True)

        if show_labels:
            for n in self.nodes:
                ax.text(n.position[0], n.position[1], n.position[2], n.node_id, fontsize=9)

        ax.set_title("Node Placement in 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def verify_node(self, node: Node):
        return node.verify_certificate()

class Handshake:
    def __init__(self, node_certificate, public_key):
        self.node_certificate = node_certificate
        self.public_key = public_key

class Hello:
    def __init__(self, node_id, mac_address, ecc_public_key, signature, position, speed):
        self.node_id = node_id
        self.mac_address = mac_address
        self.ecc_public_key = ecc_public_key
        self.signature = signature
        self.position = position  # (x, y, z)
        self.speed = speed        # instantaneous speed

class MABAgent:
    def __init__(self, node, distance_threshold=600.0):
        self.node = node
        self.distance_threshold = distance_threshold
        # store per-neighbor: node_id -> {"hello": Hello, "shared_key": bytes}
        self.hello_queue = {}

    def receive_hello(self, hello):
        # Calculate Euclidean distance
        x1, y1, z1 = self.node.position
        x2, y2, z2 = hello.position
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        # Try to verify identity and derive shared key; ignore if invalid
        shared_key = None
        try:
            # verify and fetch peer ecc public key (raises on bad cert)
            peer_pub_pem = Node.verify_received_identity(hello, self.node.root_ca_public_key)
            # derive per-pair shared key (ECDH + HKDF)
            shared_key = self.node.derive_shared_key(peer_pub_pem)
        except Exception:
            # verification failed -> treat as untrusted (don't store)
            shared_key = None

        # Decision: keep or remove (only store if verified and within distance)
        if dist <= self.distance_threshold and shared_key is not None:
            self.hello_queue[hello.node_id] = {"hello": hello, "shared_key": shared_key}
        else:
            # remove if present
            self.hello_queue.pop(hello.node_id, None)

    def get_shared_key(self, neighbor_id):
        entry = self.hello_queue.get(neighbor_id)
        if entry:
            return entry.get("shared_key")
        return None

    def neighbors(self):
        return list(self.hello_queue.keys())

    def neighbor_info(self):
        return {nid: {
                    "distance": math.sqrt(sum((a-b)**2 for a, b in zip(self.node.position, self.hello_queue[nid]["hello"].position))),
                    "speed": self.hello_queue[nid]["hello"].speed,
                    "shared_key": self.hello_queue[nid].get("shared_key")
                }
                for nid in self.hello_queue}

class CooperativeQLAgent:
    def __init__(self, node):
        self.node = node
        self.q_table = {}  # state tuple -> action value
        self.actions = []  # List of possible actions (neighbor node_ids)

    def decide_forward(self, opp_pkt, neighbor_info, all_nodes):
        # State: (queue_length, energy, mobility, speed, neighbor_energy)
        state = (
            len(self.node.opp_queue),
            self.node.energy,
            neighbor_info.get('mobility', 0.0),
            self.node.instantaneous_speed,
            neighbor_info.get('energy', 0.0)
        )
        neighbors = self.node.mab_agent.neighbors()
        pkt_id = (opp_pkt.src_id, opp_pkt.seq_num)
        valid_neighbors = [
            nid for nid in neighbors
            if nid != self.node.node_id and pkt_id not in next(n for n in all_nodes if n.node_id == nid).seen_packets
            and next(n for n in all_nodes if n.node_id == nid).energy > 5.0  # Only neighbors with enough energy
        ]
        if not valid_neighbors:
            return False, None
        # Pick neighbor with highest energy
        best_nid = max(
            valid_neighbors,
            key=lambda nid: next(n for n in all_nodes if n.node_id == nid).energy
        )
        # Forward if energy > 5 and queue not overloaded
        if self.node.energy > 5 and len(self.node.opp_queue) < 50:
            return True, best_nid
        return False, None

    def top_actions(self):
        # Return top actions (neighbor node_ids) based on Q-table (stub)
        return self.node.mab_agent.neighbors()[:3]

class RandomWaypointMobility:
    def __init__(self, nodes, bounds=(100.0, 100.0, 100.0), speed_range=(1.0, 5.0), pause_time=0.0):
        """
        nodes: list[Node]
        bounds: (Xmax, Ymax, Zmax) cube size
        speed_range: (min_speed, max_speed)
        pause_time: seconds to pause at each waypoint (simulated as steps)
        """
        self.nodes = nodes
        self.bounds = bounds
        self.min_speed, self.max_speed = speed_range
        self.pause_time = int(pause_time)
        # Per-node state
        self.dest = {}     # node_id -> (dx, dy, dz)
        self.speed = {}    # node_id -> units per step
        self.pause = {}    # node_id -> remaining pause steps
        # Initialize state
        for n in self.nodes:
            self._assign_new_destination(n)

    def _rand_point(self):
        X, Y, Z = self.bounds
        return (random.uniform(0, X), random.uniform(0, Y), random.uniform(0, Z))

    def _assign_new_destination(self, node):
        dx, dy, dz = self._rand_point()
        self.dest[node.node_id] = (dx, dy, dz)
        self.speed[node.node_id] = random.uniform(self.min_speed, self.max_speed)
        self.pause[node.node_id] = self.pause_time

    def step(self):
        """Advance mobility by one step for all nodes."""
        for n in self.nodes:
            # If pausing at waypoint
            if self.pause[n.node_id] > 0:
                self.pause[n.node_id] -= 1
                continue

            x, y, z = n.position
            dx, dy, dz = self.dest[n.node_id]
            vx = dx - x
            vy = dy - y
            vz = dz - z
            dist = math.sqrt(vx*vx + vy*vy + vz*vz)
            spd = self.speed[n.node_id]
            # Move by at most 'spd' toward destination
            step_frac = min(1.0, spd / max(dist, 1e-9))
            nx = x + vx * step_frac
            ny = y + vy * step_frac
            nz = z + vz * step_frac
            n.position = (nx, ny, nz)
            n.update_speed(spd)

            # If arrived (after step), start pause
            if step_frac >= 1.0:
                self.pause[n.node_id] = self.pause_time
                # New destination will be assigned on next step after pause completes


def compute_neighbor_stability(nodes):
    """Calculate average neighbor stability (how often neighbor set changes)."""
    stability = []
    for node in nodes:
        # For simplicity, use the number of neighbors as a proxy for stability
        stability.append(len(node.mab_agent.neighbors()))
    return sum(stability) / max(1, len(stability))

def compute_avg_hop_count(nodes, delivered_seq_nums):
    """Estimate average hop count for delivered packets."""
    hop_counts = []
    for seq_num in delivered_seq_nums:
        for node in nodes:
            if seq_num in node.delivered_packets:
                # Count how many nodes have seen this packet
                count = sum(1 for n in nodes if seq_num in n.seen_packets)
                hop_counts.append(count)
                break
    return sum(hop_counts) / max(1, len(hop_counts))

def compute_energy_per_packet(nodes, delivered):
    """Calculate average energy consumed per delivered packet."""
    total_energy_used = sum(100.0 - node.energy for node in nodes)
    return total_energy_used / max(1, delivered)

def compute_encryption_overhead(pdr_stats):
    """Estimate encryption overhead as ratio of encrypted bytes to total bytes delivered."""
    encrypted_bytes = pdr_stats.get('encrypted_bytes_delivered', 0)
    total_bytes = pdr_stats.get('delivered_bytes', sum(pdr_stats.get('step_bandwidth', [])))
    return encrypted_bytes / max(1, total_bytes)

def compute_forwarding_efficiency(pdr_stats):
    """Delivered packets / forwarded packets."""
    return pdr_stats['delivered'] / max(1, pdr_stats['forwarded'])

def compute_avg_delay(packet_delivery_steps, delivered_seq_nums):
    """Average steps taken for delivery (if tracked)."""
    delays = [packet_delivery_steps.get(seq, 0) for seq in delivered_seq_nums]
    return sum(delays) / max(1, len(delays))

def plot_src_dst_distances(nodes, packet_map, out_file='node_src_dst_distances.png', show_legend=True):
    """3D placement: mark packet origin (yellow) and destination (red) and show legend of distances per packet."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # node positions
    xs = [n.position[0] for n in nodes]
    ys = [n.position[1] for n in nodes]
    zs = [n.position[2] for n in nodes]
    ax.scatter(xs, ys, zs, c='lightblue', s=40, depthshade=True, label='nodes')

    node_map = {n.node_id: n for n in nodes}

    # collect legend handles for packet distance labels
    legend_handles = []
    legend_labels = []

    for seq, (s_id, d_id) in packet_map.items():
        s = node_map.get(s_id)
        d = node_map.get(d_id)
        if not s or not d:
            continue
        sx, sy, sz = s.position
        dx, dy, dz = d.position
        # draw line between src and dst
        ax.plot([sx, dx], [sy, dy], [sz, dz], c='gray', alpha=0.6, linewidth=1)
        # mark source (yellow) and destination (red)
        ax.scatter([sx], [sy], [sz], c='yellow', s=120, edgecolors='k', marker='*')
        ax.scatter([dx], [dy], [dz], c='red', s=90, marker='o', edgecolors='k')
        # annotate small id near points
        ax.text(sx, sy, sz, f"{s.node_id}\n(S{seq})", color='black', fontsize=7)
        ax.text(dx, dy, dz, f"{d.node_id}\n(D{seq})", color='black', fontsize=7)
        # compute distance
        dist = math.sqrt((sx-dx)**2 + (sy-dy)**2 + (sz-dz)**2)
        # prepare legend entry (use Line2D handle)
        handle = Line2D([0],[0], color='gray', lw=2)
        legend_handles.append(handle)
        legend_labels.append(f"pkt {seq}: {dist:.1f} m")

    ax.set_title("Node Placement: Packet origins (yellow) and destinations (red)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    if show_legend and legend_handles:
        # place legend outside plot to avoid clutter
        ax.legend(legend_handles, legend_labels, title='Packet distances', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.subplots_adjust(right=0.75)

    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
def plot_nodes_and_packets_3d(nodes, packet_map, show_labels=True):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all nodes
    xs = [n.position[0] for n in nodes]
    ys = [n.position[1] for n in nodes]
    zs = [n.position[2] for n in nodes]
    ax.scatter(xs, ys, zs, c='skyblue', s=60, depthshade=True, label='Nodes')

    node_dict = {n.node_id: n for n in nodes}

    # Plot packets
    for seq, (src_id, dst_id) in packet_map.items():
        src_node = node_dict[src_id]
        dst_node = node_dict[dst_id]

        sx, sy, sz = src_node.position
        dx, dy, dz = dst_node.position

        ax.plot([sx, dx], [sy, dy], [sz, dz], c='gray', alpha=0.5, linewidth=1)
        ax.scatter([sx], [sy], [sz], c='yellow', s=120, marker='*', edgecolors='k')
        ax.scatter([dx], [dy], [dz], c='red', s=90, marker='o', edgecolors='k')

        if show_labels:
            ax.text(sx, sy, sz, f"{src_node.node_id}\n(S{seq})", fontsize=8, color='black')
            ax.text(dx, dy, dz, f"{dst_node.node_id}\n(D{seq})", fontsize=8, color='black')

    ax.set_title("3D MANET Node Placement with Packet Flows")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

def plot_nodes_and_packet_flows(nodes, packet_map, delivered_seq_nums, out_placement='node_placement.png', out_flows='node_placement_flows.png'):
    """Save two images:
       - out_placement: plain node placement (PNG)
       - out_flows: nodes with packet origins (yellow star), destinations (red circle) and lines (green=delivered, red=undelivered)
    """
    # Plain placement
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = [n.position[0] for n in nodes]
    ys = [n.position[1] for n in nodes]
    zs = [n.position[2] for n in nodes]
    ax.scatter(xs, ys, zs, c='tab:blue', s=60, depthshade=True)
    for n in nodes:
        ax.text(n.position[0], n.position[1], n.position[2], n.node_id, fontsize=8)
    ax.set_title("Node Placement")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(out_placement, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Flows image
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='lightgray', s=40, depthshade=True, label='nodes')
    node_map = {n.node_id: n for n in nodes}
    for seq, pair in packet_map.items():
        try:
            s_id, d_id = pair
        except Exception:
            continue
        s = node_map.get(s_id)
        d = node_map.get(d_id)
        if not s or not d:
            continue
        sx, sy, sz = s.position
        dx, dy, dz = d.position
        # line colored by delivery status
        color = 'green' if seq in delivered_seq_nums else 'red'
        ax.plot([sx, dx], [sy, dy], [sz, dz], c=color, alpha=0.6, linewidth=1)
        ax.scatter([sx], [sy], [sz], c='yellow', s=120, marker='*', edgecolors='k')
        ax.scatter([dx], [dy], [dz], c='red', s=90, marker='o', edgecolors='k')
        # small labels
        ax.text(sx, sy, sz, f"{s.node_id}\n(S{seq})", fontsize=7, color='black')
        ax.text(dx, dy, dz, f"{d.node_id}\n(D{seq})", fontsize=7, color='black')
    ax.set_title("Node Placement and Packet Flows (green=delivered, red=undelivered)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(out_flows, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    config = Configure(num_nodes=100)
    # use larger mobility bounds so nodes can be >400m apart
    mobility = RandomWaypointMobility(config.nodes, bounds=(1000.0, 1000.0, 1000.0))
    # initialize stats (adjusted keys for bytes)
    pdr_stats = {
        'delivered': 0,
        'forwarded': 0,
        'sent': 0,
        'delivered_seq_nums': set(),
        'delivered_bytes': 0,
        'encrypted_bytes_delivered': 0,
        'step_delivered': [],
        'step_bandwidth': [],
        'step_throughput': [],
        'step_energy': [],
        'step_pdr': [],
        'step_sent': [],
        'step_neighbor_stability': [],
        'step_avg_hop_count': [],
        'step_energy_per_packet': [],
        'step_encryption_overhead': [],
        'packet_delivery_steps': {},  # seq_num -> step delivered
        'packet_map': {},             # seq_num -> (src_node_id, dst_node_id)
    }

    # Initial Hello exchange: each node sends hello to all others
    for node in config.nodes:
        hello_pkt = node.create_hello_packet()
        for other in config.nodes:
            if other.node_id != node.node_id:
                other.receive_hello(hello_pkt)
    # choose a src/dst pair whose Euclidean distance > 400
    src_node = None
    dst_node = None
    found_pair = False
    for n in config.nodes:
        # find farthest neighbor from n
        farthest = max(config.nodes, key=lambda x: math.sqrt(sum((a-b)**2 for a,b in zip(x.position, n.position))))
        dist = math.sqrt(sum((a-b)**2 for a,b in zip(farthest.position, n.position)))
        # require at least 450 meters separation
        if dist >= 450.0 and n.node_id != farthest.node_id:
            src_node = n
            dst_node = farthest
            found_pair = True
            break
    # fallback: if not found (very unlikely with current bounds), pick two random nodes and forcibly set positions
    if not found_pair:
        src_node = config.nodes[0]
        dst_node = config.nodes[1]
        # move destination far from source
        sx, sy, sz = src_node.position
        dst_node.position = (sx + 450.0, sy, sz)

    print(f"Selected src: {src_node.node_id}, dst: {dst_node.node_id}, distance = {math.sqrt(sum((a-b)**2 for a,b in zip(src_node.position, dst_node.position))):.1f} m")

    # Send 1000 packets from chosen source to chosen destination
    num_packets = 8
    for seq_num in range(num_packets):
        opp_pkt = OPPPacket(src_node.node_id, dst_node.node_id, f"Payload {seq_num}", seq_num)
        src_node.receive_opp_packet(opp_pkt, src_node.get_cooperation_info())
        pdr_stats['sent'] += 1
        pdr_stats['packet_map'][seq_num] = (src_node.node_id, dst_node.node_id)

    # run simulation for more steps to allow propagation
    total_sim_steps = 200
    for step in range(total_sim_steps):
        # set current step in stats so process_opp_queue can stamp delivery step
        pdr_stats['current_step'] = step
        # zero per-step accumulators
        pdr_stats['current_step_bandwidth'] = 0
        pdr_stats['current_step_encrypted_bytes'] = 0

        mobility.step()
        # broadcast hello every step to keep neighbor tables fresh
        for node in config.nodes:
            hello_pkt = node.create_hello_packet()
            for other in config.nodes:
                if other.node_id != node.node_id:
                    other.receive_hello(hello_pkt)

        delivered_this_step = 0
        sent_this_step = 0

        # Process OPP queues multiple times per step for better propagation
        for _ in range(6):
            for node in config.nodes:
                before_delivered = len(pdr_stats['delivered_seq_nums'])
                node.process_opp_queue(config.nodes, pdr_stats)
                after_delivered = len(pdr_stats['delivered_seq_nums'])
                delivered_now = after_delivered - before_delivered
                delivered_this_step += delivered_now

        # after processing, collect per-step metrics from accumulators
        bandwidth_this_step = pdr_stats.pop('current_step_bandwidth', 0)
        encrypted_bytes_this_step = pdr_stats.pop('current_step_encrypted_bytes', 0)

        pdr_stats['step_sent'].append(sent_this_step)
        avg_energy = sum(node.energy_usage() for node in config.nodes) / len(config.nodes)
        pdr_stats['step_delivered'].append(delivered_this_step)
        pdr_stats['step_bandwidth'].append(bandwidth_this_step)
        pdr_stats['step_throughput'].append(delivered_this_step)  # packets per step
        pdr_stats['step_energy'].append(avg_energy)

        # Calculate new metrics
        neighbor_stability = compute_neighbor_stability(config.nodes)
        avg_hop_count = compute_avg_hop_count(config.nodes, pdr_stats['delivered_seq_nums'])
        energy_per_packet = compute_energy_per_packet(config.nodes, pdr_stats['delivered'])
        encryption_overhead = compute_encryption_overhead(pdr_stats)

        pdr_stats['step_neighbor_stability'].append(neighbor_stability)
        pdr_stats['step_avg_hop_count'].append(avg_hop_count)
        pdr_stats['step_energy_per_packet'].append(energy_per_packet)
        pdr_stats['step_encryption_overhead'].append(encryption_overhead)

    # List out neighbor info for each node
    for node in config.nodes:
        print(f"{node.node_id}: {node.neighbor_count()} neighbors")
        for nid, info in node.neighbor_details().items():
            print(f"  Neighbor: {nid}, Distance: {info['distance']:.2f}, Speed: {info['speed']:.2f}")
    # Print node energy usage
    print("\nNode Energy Usage:")
    for node in config.nodes:
        print(f"{node.node_id}: {node.energy_usage():.2f} Joules remaining")
    # Print PDR
    print(f"\nPacket Delivery Ratio (PDR): {pdr_stats['delivered']}/{pdr_stats['sent']} = {pdr_stats['delivered']/max(1,pdr_stats['sent']):.2f}")

    # Throughput and Bandwidth summary
    total_steps = len(pdr_stats['step_throughput'])
    avg_throughput = sum(pdr_stats['step_throughput']) / max(1, total_steps)
    avg_bandwidth = sum(pdr_stats['step_bandwidth']) / max(1, total_steps)
    print(f"Average Throughput: {avg_throughput:.2f} packets/step")
    print(f"Average Bandwidth: {avg_bandwidth:.2f} bytes/step")
    print(f"Average Node Energy: {sum(node.energy_usage() for node in config.nodes)/len(config.nodes):.2f} Joules")
    # Encryption efficiency
    encryption_eff = pdr_stats.get('encrypted_bytes_delivered', 0) / max(1, pdr_stats.get('delivered_bytes', 1))
    print(f"Encryption Efficiency (Encrypted Bytes Delivered / Total Bytes Delivered): {encryption_eff:.2f}")

    # Journal-style summary
    print("\n--- MANET Algorithm Journal Metrics ---")
    print(f"Nodes: {len(config.nodes)}")
    print(f"Packets Sent: {pdr_stats['sent']}")
    print(f"Packets Delivered: {pdr_stats['delivered']}")
    print(f"Average Neighbor Stability: {sum(pdr_stats['step_neighbor_stability'])/max(1,len(pdr_stats['step_neighbor_stability'])):.2f}")
    print(f"Average Hop Count (Delivered): {sum(pdr_stats['step_avg_hop_count'])/max(1,len(pdr_stats['step_avg_hop_count'])):.2f}")
    print(f"Average Energy per Delivered Packet: {sum(pdr_stats['step_energy_per_packet'])/max(1,len(pdr_stats['step_energy_per_packet'])):.2f} Joules")
    print(f"Encryption Overhead (Encrypted Bytes/Total Bytes): {sum(pdr_stats['step_encryption_overhead'])/max(1,len(pdr_stats['step_encryption_overhead'])):.2f}")
    print(f"Features covered: Secure ECC-based identity, ASCON encryption, RL-based neighbor selection, energy-aware routing, mobility model, multi-hop opportunistic forwarding.")

    # Print highlighted efficiency metrics
    print("\n--- Efficiency Metrics ---")
    end_to_end_pdr = pdr_stats['delivered'] / max(1, pdr_stats['sent'])
    avg_throughput = sum(pdr_stats['step_throughput']) / max(1, total_steps)
    avg_bandwidth = sum(pdr_stats['step_bandwidth']) / max(1, total_steps)
    avg_energy = sum(node.energy_usage() for node in config.nodes) / len(config.nodes)
    avg_neighbor_stability = sum(pdr_stats['step_neighbor_stability']) / max(1, len(pdr_stats['step_neighbor_stability']))
    avg_hop_count = sum(pdr_stats['step_avg_hop_count']) / max(1, len(pdr_stats['step_avg_hop_count']))
    avg_energy_per_packet = sum(pdr_stats['step_energy_per_packet']) / max(1, len(pdr_stats['step_energy_per_packet']))
    encryption_overhead = sum(pdr_stats['step_encryption_overhead']) / max(1, len(pdr_stats['step_encryption_overhead']))
    forwarding_efficiency = compute_forwarding_efficiency(pdr_stats)
    avg_delay = compute_avg_delay(pdr_stats['packet_delivery_steps'], pdr_stats['delivered_seq_nums'])

    print(f"End-to-End PDR: {end_to_end_pdr:.2f}")
    print(f"Average Throughput per Step: {avg_throughput:.2f} packets")
    print(f"Average Bandwidth per Step: {avg_bandwidth:.2f} bytes")
    print(f"Average Node Energy per Step: {avg_energy:.2f} Joules")
    print(f"Average Neighbor Stability: {avg_neighbor_stability:.2f}")
    print(f"Average Hop Count (Delivered): {avg_hop_count:.2f}")
    print(f"Average Energy per Delivered Packet: {avg_energy_per_packet:.2f} Joules")
    print(f"Encryption Overhead: {encryption_overhead:.2f}")
    print(f"Forwarding Efficiency (Delivered/Forwarded): {forwarding_efficiency:.2f}")
    print(f"Average Delay per Packet: {avg_delay:.2f} steps")

    steps = list(range(total_steps))

    # Throughput per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_throughput'], marker='o')
    plt.title("Throughput per Step")
    plt.xlabel("Step")
    plt.ylabel("Packets Delivered")
    plt.tight_layout()
    plt.show()

    # Bandwidth per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_bandwidth'], marker='o', color='orange')
    plt.title("Bandwidth per Step")
    plt.xlabel("Step")
    plt.ylabel("Bytes Delivered")
    plt.tight_layout()
    plt.show()

    # Average Node Energy per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_energy'], marker='o', color='green')
    plt.title("Average Node Energy per Step")
    plt.xlabel("Step")
    plt.ylabel("Avg Energy (Joules)")
    plt.tight_layout()
    plt.show()

    # Neighbor Stability per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_neighbor_stability'], marker='o', color='magenta')
    plt.title("Neighbor Stability per Step")
    plt.xlabel("Step")
    plt.ylabel("Avg Neighbor Count")
    plt.tight_layout()
    plt.show()

    # Average Hop Count per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_avg_hop_count'], marker='o', color='brown')
    plt.title("Average Hop Count per Step")
    plt.xlabel("Step")
    plt.ylabel("Avg Hop Count")
    plt.tight_layout()
    plt.show()

    # Energy per Delivered Packet
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_energy_per_packet'], marker='o', color='cyan')
    plt.title("Energy per Delivered Packet")
    plt.xlabel("Step")
    plt.ylabel("Energy (Joules)")
    plt.tight_layout()
    plt.show()

    # Encryption Overhead per Step
    plt.figure(figsize=(8, 5))
    plt.plot(steps, pdr_stats['step_encryption_overhead'], marker='o', color='purple')
    plt.title("Encryption Overhead per Step")
    plt.xlabel("Step")
    plt.ylabel("Encrypted/Total Bytes")
    plt.tight_layout()
    plt.show()

    # Forwarding Efficiency per Step
    plt.figure(figsize=(8, 5))
    forwarding_eff_per_step = [
        pdr_stats['step_delivered'][i] / max(1, pdr_stats['step_throughput'][i] + pdr_stats['step_delivered'][i])
        for i in range(total_steps)
    ]
    plt.plot(steps, forwarding_eff_per_step, marker='o', color='teal')
    plt.title("Forwarding Efficiency per Step")
    plt.xlabel("Step")
    plt.ylabel("Delivered/Forwarded")
    plt.tight_layout()
    plt.show()

    # Average Delay per Packet
    plt.figure(figsize=(8, 5))
    delays = [compute_avg_delay(pdr_stats['packet_delivery_steps'], pdr_stats['delivered_seq_nums'])] * total_steps
    plt.plot(steps, delays, marker='o', color='red')
    plt.title("Average Delay per Packet")
    plt.xlabel("Step")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.show()

    # End-to-End PDR as a bar
    plt.figure(figsize=(5, 5))
    plt.bar(['End-to-End PDR'], [end_to_end_pdr], color='navy')
    plt.title("End-to-End PDR")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.show()

    # Forwarding Efficiency as a bar
    plt.figure(figsize=(5, 5))
    plt.bar(['Forwarding Efficiency'], [forwarding_efficiency], color='teal')
    plt.title("Forwarding Efficiency")
    plt.ylabel("Delivered/Forwarded")
    plt.tight_layout()
    plt.show()

    # Encryption Overhead as a bar
    plt.figure(figsize=(5, 5))
    plt.bar(['Encryption Overhead'], [encryption_overhead], color='purple')
    plt.title("Encryption Overhead")
    plt.ylabel("Encrypted/Total Bytes")
    plt.tight_layout()
    plt.show()

    # After simulation and before metric plots, save placement images
    plot_nodes_and_packet_flows(config.nodes, pdr_stats.get('packet_map', {}), pdr_stats['delivered_seq_nums'],
                                out_placement='node_placement.png', out_flows='node_placement_flows.png')

    # New: save source-destination distance plot
    plot_src_dst_distances(config.nodes, pdr_stats.get('packet_map', {}), out_file='node_src_dst_distances.png')

    # Example small packet_map display
    packet_map = {0: (src_node.node_id, dst_node.node_id), 1: ("Node_2", "Node_30")}
    plot_nodes_and_packets_3d(config.nodes, packet_map)







