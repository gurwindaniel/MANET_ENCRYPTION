# ...existing code...
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature
import os

class Node:
    def __init__(self, node_id, ip_address, mac_address, root_ca_public_key, node_certificate, ecc_public_key, ecc_private_key):
        self.node_id = node_id
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.root_ca_public_key = root_ca_public_key          # PEM string
        self.node_certificate = node_certificate              # hex signature over id+mac+pubkey
        self.ecc_public_key = ecc_public_key                  # PEM string
        self.ecc_private_key = ecc_private_key                # PEM string

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

if __name__ == "__main__":
    config = Configure(num_nodes=3)
    hello_node0=config.nodes[0].create_hello_packet()
    key1=config.nodes[1].verify_received_identity(hello_node0,config.nodes[1].root_ca_public_key)
    key2=config.nodes[0].ecc_public_key

    


    

