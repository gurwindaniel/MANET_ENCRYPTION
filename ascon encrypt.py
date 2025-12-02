# ...existing code...
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import math

class Node:
    def __init__(self, node_id, ip_address, mac_address, root_ca_public_key, node_certificate, ecc_public_key, ecc_private_key):
        self.node_id = node_id
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.root_ca_public_key = root_ca_public_key          # PEM string
        self.node_certificate = node_certificate              # hex signature over id+mac+pubkey
        self.ecc_public_key = ecc_public_key                  # PEM string
        self.ecc_private_key = ecc_private_key  
        self.position = (0.0, 0.0, 0.0)              # PEM string

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
        signature = self.node_certificate  # Assuming the certificate is used as a signature for simplicity
        return Hello(node_id=self.node_id, mac_address=self.mac_address, ecc_public_key=self.ecc_public_key, signature=signature)

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
        # assign random 3D positions for each node
        self._assign_random_positions(space_size=100.0)

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
# ...existing code..

    def verify_node(self, node: Node):
        return node.verify_certificate()

class Handshake:
    def __init__(self, node_certificate, public_key):
        self.node_certificate = node_certificate
        self.public_key = public_key

class Hello:
    def __init__(self, node_id, mac_address, ecc_public_key, signature):
        self.node_id = node_id
        self.mac_address = mac_address
        self.ecc_public_key = ecc_public_key
        self.signature = signature
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

            if dist < 1e-6:
                # Reached destination: assign a new one and start pause
                self._assign_new_destination(n)
                continue

            spd = self.speed[n.node_id]
            # Move by at most 'spd' toward destination
            step_frac = min(1.0, spd / max(dist, 1e-9))
            nx = x + vx * step_frac
            ny = y + vy * step_frac
            nz = z + vz * step_frac
            n.position = (nx, ny, nz)

            # If arrived (after step), start pause
            if step_frac >= 1.0:
                self.pause[n.node_id] = self.pause_time
                # New destination will be assigned on next step after pause completes


if __name__ == "__main__":
    config = Configure(num_nodes=100)
    hello_node0=config.nodes[0].create_hello_packet()
    key1=config.nodes[1].verify_received_identity(hello_node0,config.nodes[1].root_ca_public_key)
    key2=config.nodes[0].ecc_public_key
    # Plot the 3D positions of nodes
    config.plot_nodes_3d(show_labels=True)
    mobility = RandomWaypointMobility(config.nodes)
    config.plot_nodes_3d(show_labels=True)
    for step in range(10):
        mobility.step()
        config.plot_nodes_3d(show_labels=False)

    


    

