# -*- coding: utf-8 -*-
import os
import random
import math
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature

try:
    import ascon
except ImportError:
    ascon = None

logging.basicConfig(level=logging.INFO, format='%(message)s')


# -------------------- Packet Forwarding --------------------
def forward_packet(packet, next_hop, pdr_stats):
    """Forward a packet to a neighbor and track metrics."""
    packet.hop_count += 1
    pdr_stats['forwarded'] += 1
    pdr_stats['packet_hop_counts'][packet.seq_num] = packet.hop_count
    pdr_stats['packet_energy_used'][packet.seq_num] = pdr_stats.get('packet_energy_used', {}).get(packet.seq_num, 0) + 0.05
    logging.info(f"FORWARD: Packet {packet.seq_num} forwarded to {next_hop.node_id}, hop_count={packet.hop_count}")
    next_hop.receive_opp_packet(packet, next_hop.get_cooperation_info())


def deliver_packet(packet, node, pdr_stats, step):
    """Deliver a packet to destination and track metrics."""
    if not packet.delivered:
        packet.delivered = True
        node.delivered_packets.add(packet.seq_num)
        if packet.seq_num not in pdr_stats.get('delivered_seq_nums', set()):
            pdr_stats['delivered'] += 1
            pdr_stats.setdefault('delivered_seq_nums', set()).add(packet.seq_num)
            pdr_stats['packet_hop_counts'][packet.seq_num] = packet.hop_count
            pdr_stats['packet_energy_used'][packet.seq_num] = pdr_stats.get('packet_energy_used', {}).get(packet.seq_num, 0)
            pdr_stats['packet_delivery_steps'][packet.seq_num] = step
            logging.info(f"DELIVER: Packet {packet.seq_num} delivered at node {node.node_id}, hop_count={packet.hop_count}, energy_used={pdr_stats['packet_energy_used'][packet.seq_num]:.2f}")


# -------------------- Packet Class --------------------
class OPPPacket:
    def __init__(self, src_id, dst_id, payload, seq_num):
        self.src_id = src_id
        self.dst_id = dst_id
        self.payload = payload
        self.seq_num = seq_num
        self.delivered = False
        self.hop_count = 0


# -------------------- Node Class --------------------
class Node:
    def __init__(self, node_id, ip_address, mac_address, root_ca_public_key, node_certificate, ecc_public_key, ecc_private_key):
        self.node_id = node_id
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.root_ca_public_key = root_ca_public_key
        self.node_certificate = node_certificate
        self.ecc_public_key = ecc_public_key
        self.ecc_private_key = ecc_private_key
        self.position = (0.0, 0.0, 0.0)
        self.instantaneous_speed = 0.0
        self.energy = 100.0
        self.mab_agent = MABAgent(self)
        self.opp_queue = []
        self.q_agent = CooperativeQLAgent(self)
        self.delivered_packets = set()
        self.seen_packets = set()
        self.forwarded_seq_nums = set()

    def __repr__(self):
        return f"Node({self.node_id})"

    def verify_certificate(self):
        data = f"{self.node_id}:{self.mac_address}:{self.ecc_public_key}".encode()
        signature = bytes.fromhex(self.node_certificate)
        ca_pub = serialization.load_pem_public_key(self.root_ca_public_key.encode())
        try:
            ca_pub.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False

    def export_identity(self):
        return {
            "node_id": self.node_id,
            "mac_address": self.mac_address,
            "certificate": self.node_certificate,
            "ecc_public_key": self.ecc_public_key,
        }

    def create_hello_packet(self):
        if self.energy > 5.0:
            self.drain_energy(0.01)
        return Hello(
            node_id=self.node_id,
            mac_address=self.mac_address,
            ecc_public_key=self.ecc_public_key,
            signature=self.node_certificate,
            position=self.position,
            speed=self.instantaneous_speed
        )

    def receive_hello(self, hello):
        if self.energy > 5.0:
            self.drain_energy(0.005)
        self.mab_agent.receive_hello(hello)

    def drain_energy(self, amount):
        self.energy = max(0.0, self.energy - amount)

    def energy_usage(self):
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

    def process_opp_queue(self, all_nodes, pdr_stats, step):
        new_queue = []
        max_forwards_per_packet = 2
        for opp_pkt, neighbor_info in self.opp_queue:
            pkt_id = (opp_pkt.src_id, opp_pkt.seq_num)
            if opp_pkt.dst_id == self.node_id:
                deliver_packet(opp_pkt, self, pdr_stats, step)
                continue

            dst_node = next((n for n in all_nodes if n.node_id == opp_pkt.dst_id), None)
            if not dst_node:
                continue

            my_dist = math.sqrt(sum((a-b)**2 for a, b in zip(self.position, dst_node.position)))
            forwarded = False
            valid_neighbors = []
            for nid in self.mab_agent.neighbors():
                if nid == self.node_id:
                    continue
                neighbor = next((n for n in all_nodes if n.node_id == nid), None)
                if not neighbor or neighbor.energy <= 5.0:
                    continue
                neighbor_dist = math.sqrt(sum((a-b)**2 for a, b in zip(neighbor.position, dst_node.position)))
                if pkt_id not in neighbor.seen_packets:
                    valid_neighbors.append((neighbor, neighbor_dist))
            valid_neighbors.sort(key=lambda x: x[0].energy, reverse=True)
            closer_neighbors = [n for n, d in valid_neighbors if d < my_dist]
            targets = closer_neighbors if closer_neighbors else [n for n, _ in valid_neighbors]

            if targets and self.energy > 5.0 and pkt_id not in self.forwarded_seq_nums:
                forwards = 0
                for neighbor in targets[:max_forwards_per_packet]:
                    key = neighbor.get_cooperation_info().get('shared_key', b'1234567890123456')
                    payload_bytes = opp_pkt.payload.encode() if isinstance(opp_pkt.payload, str) else opp_pkt.payload
                    encrypted_payload = self.encrypt_payload_bytes(payload_bytes, key)
                    new_pkt = OPPPacket(self.node_id, opp_pkt.dst_id, encrypted_payload, opp_pkt.seq_num)
                    new_pkt.hop_count = opp_pkt.hop_count
                    forward_packet(new_pkt, neighbor, pdr_stats)
                    self.drain_energy(0.05)
                    forwarded = True
                    forwards += 1
                    if forwards >= max_forwards_per_packet:
                        break
                self.forwarded_seq_nums.add(pkt_id)
            if not forwarded:
                new_queue.append((opp_pkt, neighbor_info))
        self.opp_queue = new_queue

    def encrypt_payload_bytes(self, payload_bytes, key):
        if ascon:
            nonce = b'0000000000000000'
            key = key if isinstance(key, bytes) else bytes(key, 'utf-8')
            key = key.ljust(16, b'\0')[:16]
            payload_bytes = payload_bytes if isinstance(payload_bytes, bytes) else bytes(payload_bytes)
            try:
                return ascon.encrypt(key, nonce, payload_bytes, b'')
            except Exception:
                return payload_bytes
        return payload_bytes

    def get_cooperation_info(self):
        return {
            'mobility': self.instantaneous_speed,
            'energy': self.energy,
            'stability': 1.0,
            'q_top_actions': self.q_agent.top_actions(),
            'shared_key': b'1234567890123456',
        }


# -------------------- Supporting Classes --------------------
class Hello:
    def __init__(self, node_id, mac_address, ecc_public_key, signature, position, speed):
        self.node_id = node_id
        self.mac_address = mac_address
        self.ecc_public_key = ecc_public_key
        self.signature = signature
        self.position = position
        self.speed = speed


class MABAgent:
    def __init__(self, node, distance_threshold=250.0):
        self.node = node
        self.distance_threshold = distance_threshold
        self.hello_queue = {}

    def receive_hello(self, hello):
        x1, y1, z1 = self.node.position
        x2, y2, z2 = hello.position
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        if dist <= self.distance_threshold:
            self.hello_queue[hello.node_id] = hello
        else:
            self.hello_queue.pop(hello.node_id, None)

    def neighbors(self):
        return list(self.hello_queue.keys())

    def neighbor_info(self):
        return {nid: {"distance": math.sqrt(sum((a-b)**2 for a, b in zip(self.node.position, self.hello_queue[nid].position))),
                      "speed": self.hello_queue[nid].speed}
                for nid in self.hello_queue}


class CooperativeQLAgent:
    def __init__(self, node):
        self.node = node

    def top_actions(self):
        return self.node.mab_agent.neighbors()[:3]


# -------------------- Mobility --------------------
class RandomWaypointMobility:
    def __init__(self, nodes, bounds=(100.0, 100.0, 100.0), speed_range=(1.0, 5.0), pause_time=0):
        self.nodes = nodes
        self.bounds = bounds
        self.min_speed, self.max_speed = speed_range
        self.pause_time = pause_time
        self.dest = {}
        self.speed = {}
        self.pause = {}
        for n in self.nodes:
            self._assign_new_destination(n)

    def _rand_point(self):
        X, Y, Z = self.bounds
        return (random.uniform(0, X), random.uniform(0, Y), random.uniform(0, Z))

    def _assign_new_destination(self, node):
        self.dest[node.node_id] = self._rand_point()
        self.speed[node.node_id] = random.uniform(self.min_speed, self.max_speed)
        self.pause[node.node_id] = self.pause_time

    def step(self):
        for n in self.nodes:
            if self.pause[n.node_id] > 0:
                self.pause[n.node_id] -= 1
                continue
            x, y, z = n.position
            dx, dy, dz = self.dest[n.node_id]
            vx, vy, vz = dx-x, dy-y, dz-z
            dist = math.sqrt(vx*vx + vy*vy + vz*vz)
            spd = self.speed[n.node_id]
            step_frac = min(1.0, spd / max(dist, 1e-9))
            n.position = (x+vx*step_frac, y+vy*step_frac, z+vz*step_frac)
            n.update_speed(spd)
            if step_frac >= 1.0:
                self.pause[n.node_id] = self.pause_time
                self._assign_new_destination(n)


# -------------------- Metrics --------------------
def compute_neighbor_stability(nodes):
    return sum(len(n.mab_agent.neighbors()) for n in nodes) / max(1, len(nodes))


def compute_avg_hop_count(nodes, delivered_seq_nums, packet_hop_counts):
    hop_counts = [packet_hop_counts.get(seq, 0) for seq in delivered_seq_nums]
    return sum(hop_counts)/max(1, len(hop_counts))


def compute_energy_per_packet(nodes, delivered, packet_energy_used, delivered_seq_nums):
    energies = [packet_energy_used.get(seq, 0) for seq in delivered_seq_nums]
    return sum(energies)/max(1, delivered)


def compute_forwarding_efficiency(pdr_stats):
    return pdr_stats['delivered']/max(1, pdr_stats['forwarded'])


def compute_avg_delay(packet_delivery_steps, delivered_seq_nums):
    delays = [packet_delivery_steps.get(seq, 0) for seq in delivered_seq_nums]
    return sum(delays)/max(1, len(delays))


def compute_encryption_overhead(pdr_stats):
    total_bytes = sum(pdr_stats['step_bandwidth'])
    encrypted_bytes = pdr_stats['encrypted_delivered']
    return encrypted_bytes/max(1, total_bytes)


# -------------------- Network Configuration --------------------
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
        self._assign_random_positions()

    def _generate_mac(self):
        return "02:00:%02x:%02x:%02x:%02x" % tuple(os.urandom(4))

    def _generate_ip(self, idx):
        return f"192.168.1.{idx+1}"

    def _sign_certificate(self, node_id, mac_address, ecc_public_key_pem):
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

            node = Node(node_id, ip_address, mac_address, self.root_ca_public_key,
                        cert, ecc_pub_pem, ecc_priv_pem)
            self.nodes.append(node)

    def _assign_random_positions(self, space_size=100.0):
        for n in self.nodes:
            n.position = (random.uniform(0, space_size), random.uniform(0, space_size), random.uniform(0, space_size))

    def plot_nodes_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [n.position[0] for n in self.nodes]
        ys = [n.position[1] for n in self.nodes]
        zs = [n.position[2] for n in self.nodes]
        ax.scatter(xs, ys, zs)
        for n in self.nodes:
            ax.text(n.position[0], n.position[1], n.position[2], n.node_id)
        plt.show()


# -------------------- Main Simulation --------------------
if __name__ == "__main__":
    num_nodes = 50
    config = Configure(num_nodes)
    mobility = RandomWaypointMobility(config.nodes)
    pdr_stats = {
        'delivered': 0,
        'forwarded': 0,
        'sent': 0,
        'delivered_seq_nums': set(),
        'packet_hop_counts': {},
        'packet_energy_used': {},
        'packet_delivery_steps': {},
        'step_throughput': [],
        'step_bandwidth': [],
        'step_energy': [],
        'step_neighbor_stability': [],
        'step_avg_hop_count': [],
        'step_energy_per_packet': [],
        'step_encryption_overhead': [],
        'encrypted_delivered': 0,
        'step_sent': []
    }

    # Initial Hello Exchange
    for node in config.nodes:
        hello_pkt = node.create_hello_packet()
        for other in config.nodes:
            if other.node_id != node.node_id:
                other.receive_hello(hello_pkt)

    # Send random packets
    num_packets = 50
    for seq_num in range(num_packets):
        src = random.choice(config.nodes)
        dst = random.choice([n for n in config.nodes if n.node_id != src.node_id])
        opp_pkt = OPPPacket(src.node_id, dst.node_id, f"Payload {seq_num}", seq_num)
        src.receive_opp_packet(opp_pkt, src.get_cooperation_info())
        pdr_stats['sent'] += 1

    # Mobility and packet forwarding steps
    steps = 20
    for step in range(steps):
        mobility.step()
        # Hello exchange every 2 steps
        if step % 2 == 0:
            for node in config.nodes:
                hello_pkt = node.create_hello_packet()
                for other in config.nodes:
                    if other.node_id != node.node_id:
                        other.receive_hello(hello_pkt)
        # Process queues
        for _ in range(3):
            for node in config.nodes:
                node.process_opp_queue(config.nodes, pdr_stats, step)

        # Track metrics
        avg_energy = sum(n.energy_usage() for n in config.nodes) / len(config.nodes)
        pdr_stats['step_throughput'].append(len(pdr_stats['delivered_seq_nums']))
        pdr_stats['step_bandwidth'].append(0)
        pdr_stats['step_energy'].append(avg_energy)
        pdr_stats['step_neighbor_stability'].append(compute_neighbor_stability(config.nodes))
        pdr_stats['step_avg_hop_count'].append(compute_avg_hop_count(config.nodes, pdr_stats['delivered_seq_nums'], pdr_stats['packet_hop_counts']))
        pdr_stats['step_energy_per_packet'].append(compute_energy_per_packet(config.nodes, pdr_stats['delivered'], pdr_stats['packet_energy_used'], pdr_stats['delivered_seq_nums']))
        pdr_stats['step_encryption_overhead'].append(compute_encryption_overhead(pdr_stats))

    # Print results
    print("\n--- MANET Efficiency Metrics ---")
    print(f"Packets Sent: {pdr_stats['sent']}")
    print(f"Packets Delivered: {pdr_stats['delivered']}")
    print(f"Packets Forwarded: {pdr_stats['forwarded']}")
    print(f"End-to-End PDR: {pdr_stats['delivered']/max(1, pdr_stats['sent']):.2f}")
    print(f"Average Hop Count: {compute_avg_hop_count(config.nodes, pdr_stats['delivered_seq_nums'], pdr_stats['packet_hop_counts']):.2f}")
    print(f"Average Energy per Packet: {compute_energy_per_packet(config.nodes, pdr_stats['delivered'], pdr_stats['packet_energy_used'], pdr_stats['delivered_seq_nums']):.2f} J")
    print(f"Forwarding Efficiency: {compute_forwarding_efficiency(pdr_stats):.2f}")
    print(f"Average Delay: {compute_avg_delay(pdr_stats['packet_delivery_steps'], pdr_stats['delivered_seq_nums']):.2f} steps")
    print(f"Encryption Overhead: {compute_encryption_overhead(pdr_stats):.2f}")
    print(f"Average Node Energy: {sum(n.energy_usage() for n in config.nodes)/len(config.nodes):.2f} J")