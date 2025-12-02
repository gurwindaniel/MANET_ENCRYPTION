import random
import math
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import ascon
import base64 # Import base64 for encoding/decoding signatures
from collections import deque # For position history

# Existing MABArm class
class MABArm:
    def __init__(self):
        self.plays = 0
        self.rewards = 0.0
        self._average_reward = 0.0

    @property
    def average_reward(self):
        if self.plays == 0:
            return 0.0
        return self.rewards / self.plays

    def __repr__(self):
        return f"MABArm(Plays: {self.plays}, Rewards: {self.rewards:.2f}, AvgReward: {self.average_reward:.2f})"

# OppDataPacket class (updated to include SARSA state and action)
class OppDataPacket:
    def __init__(self, source_ip, destination_ip, ttl, payload=None, packet_id=None):
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.ttl = ttl
        self.payload = payload if payload is not None else f"Hello from {source_ip}"
        self.packet_id = packet_id if packet_id is not None else random.randint(0, 2**32 - 1) # Unique ID for the packet
        self.sarsa_prev_state = None # New: To store the state when this packet was last processed for SARSA
        self.sarsa_prev_action = None # New: To store the action taken when this packet was last processed for SARSA

    def __repr__(self):
        return f"OppDataPacket(ID:{self.packet_id}, Src: {self.source_ip}, Dest: {self.destination_ip}, TTL: {self.ttl})"


# Modified Node class (updated for packet sending and MAB integration, and certificate, and position history/instantaneous speed, and SARSA)
class Node:
    def __init__(self, node_id, address, mac_address, x, y, z, speed, root_ca_public_key):
        self.node_id = node_id
        self.address = address
        self.mac_address = mac_address
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.root_ca_public_key = root_ca_public_key

        # New attributes for mobility
        self.destination_x = None
        self.destination_y = None
        self.destination_z = None

        # New attribute for neighbor discovery
        self.neighbors = {}

        # New attributes for security
        self.private_key = None
        self.public_key = None
        self.symmetric_key = None

        self.generate_security_keys()

        # New attribute for MAB
        self.mab_arms = {}
        self.packet_buffer = [] # New: Buffer to store packets awaiting forwarding
        self.last_packet_id = 0 # To generate unique packet IDs from this node

        self.node_certificate = None # New: Attribute to store the signed node certificate

        # New attributes for position history and instantaneous speed
        self.position_history = deque(maxlen=4) # Stores (x, y, z, timestamp) tuples
        self.instantaneous_speed = 0.0 # Calculated from position history

        # New attributes for SARSA
        self.q_table = {} # Q-value table (state, action) -> Q-value
        self.sarsa_alpha = 0.1 # Learning rate
        self.sarsa_gamma = 0.9 # Discount factor
        self.sarsa_epsilon = 0.1 # Exploration rate

    def generate_security_keys(self):
        """Generates ECCDSA private/public key pair for the node."""
        self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self.public_key = self.private_key.public_key()

    def _get_public_key_pem(self):
        """Returns the node's public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def _load_public_key_from_pem(self, public_key_pem):
        """Loads a public key from PEM format."""
        return serialization.load_pem_public_key(
            public_key_pem.encode('utf-8'),
            backend=default_backend()
        )

    def sign_data(self, data):
        """Signs the given data using the node's private key."""
        return self.private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )

    def verify_signature(self, public_key_pem, data, signature):
        """Verifies a signature using the provided public key and data."""
        public_key = self._load_public_key_from_pem(public_key_pem)
        try:
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception as e:
            return False

    def encrypt_data_ascon(self, data):
        """Encrypts data using Ascon with the node's symmetric key. Returns ciphertext, nonce, tag."""
        if not self.symmetric_key:
            raise ValueError("Symmetric key not set for encryption.")
        nonce = random.getrandbits(128).to_bytes(16, 'big')
        ad = b''

        # Modified to capture a single return value (combined ciphertext and tag)
        combined_ct_tag = ascon.encrypt(self.symmetric_key, nonce, ad, data)

        if not isinstance(combined_ct_tag, bytes):
            # Defensive check: if ascon.encrypt returns something unexpected (like an int or tuple of 2),
            # it's an error in context of this instruction requiring a single bytes object.
            raise TypeError(f"ascon.encrypt returned unexpected type: {type(combined_ct_tag)}. Expected bytes.")

        # Split the combined result into ciphertext and tag (assuming tag is 16 bytes)
        ciphertext = combined_ct_tag[:-16]
        tag = combined_ct_tag[-16:]

        return ciphertext, nonce, tag

    def decrypt_data_ascon(self, ciphertext, nonce, tag):
        """Decrypts and authenticates data using Ascon with the node's symmetric key. Returns plaintext or raises error."""
        if not self.symmetric_key:
            raise ValueError("Symmetric key not set for decryption.")
        ad = b''
        # Correctly pass separate ciphertext and tag to ascon.decrypt
        plaintext = ascon.decrypt(self.symmetric_key, nonce, ad, ciphertext, tag)
        return plaintext

    def _set_new_waypoint(self, x_bounds, y_bounds, z_bounds):
        self.destination_x = random.uniform(x_bounds[0], x_bounds[1])
        self.destination_y = random.uniform(y_bounds[0], y_bounds[1])
        self.destination_z = random.uniform(z_bounds[0], z_bounds[1])

    def _calculate_instantaneous_speed(self):
        """Calculates and updates the node's instantaneous speed based on its position history."""
        if len(self.position_history) >= 2:
            (x1, y1, z1, t1) = self.position_history[-2]
            (x2, y2, z2, t2) = self.position_history[-1]

            distance = math.sqrt(
                (x2 - x1)**2 +
                (y2 - y1)**2 +
                (z2 - z1)**2
            )
            time_diff = t2 - t1

            if time_diff > 0:
                self.instantaneous_speed = distance / time_diff
            else:
                self.instantaneous_speed = 0.0 # No time elapsed, or division by zero
        else:
            self.instantaneous_speed = 0.0 # Not enough history to calculate

    def update_position(self, time_step, current_sim_time, x_bounds, y_bounds, z_bounds):
        # Record current position before moving
        self.position_history.append((self.x, self.y, self.z, current_sim_time))

        if self.destination_x is None:
            self._set_new_waypoint(x_bounds, y_bounds, z_bounds)

        dx = self.destination_x - self.x
        dy = self.destination_y - self.y
        dz = self.destination_z - self.z
        distance_to_dest = math.sqrt(dx**2 + dy**2 + dz**2)

        distance_can_cover = self.speed * time_step

        if distance_to_dest <= distance_can_cover:
            self.x = self.destination_x
            self.y = self.destination_y
            self.z = self.destination_z
            self.destination_x = None
            self.destination_y = None
            self.destination_z = None
        else:
            if distance_to_dest > 0:
                unit_dx = dx / distance_to_dest
                unit_dy = dy / distance_to_dest
                unit_dz = dz / distance_to_dest

                move_x = unit_dx * distance_can_cover
                move_y = unit_dy * distance_can_cover
                move_z = unit_dz * distance_can_cover

                self.x += move_x
                self.y += move_y
                self.z += move_z

        # Calculate instantaneous speed after updating position
        self._calculate_instantaneous_speed()

    def create_hello_packet(self):
        """Creates a secure HelloPacket for the current node's state."""
        hello_content = {
            "node_id": self.node_id,
            "address": self.address,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "speed": self.speed,
            "instantaneous_speed": self.instantaneous_speed # New: Include instantaneous speed
        }
        hello_bytes = json.dumps(hello_content).encode('utf-8')

        signature = self.sign_data(hello_bytes)

        signed_payload = len(signature).to_bytes(4, 'big') + signature + hello_bytes

        encrypted_payload, nonce, tag = self.encrypt_data_ascon(signed_payload)

        return HelloPacket(
            sender_node_id=self.node_id,
            sender_public_key_pem=self._get_public_key_pem(),
            encrypted_payload=encrypted_payload,
            nonce=nonce,
            tag=tag,
            sender_node_certificate=self.node_certificate, # Pass the node's certificate
            instantaneous_speed=self.instantaneous_speed # Pass instantaneous speed to HelloPacket constructor
        )

    def _get_sarsa_state(self, packet, simulation_environment_ref):
        """Defines the SARSA state representation for the current node concerning a packet and potential forwarders."""
        is_dest_neighbor = False
        destination_node = simulation_environment_ref.get_node_by_address(packet.destination_ip)
        if destination_node and destination_node.node_id in self.neighbors:
            is_dest_neighbor = True

        total_mab_reward = 0.0
        num_mab_arms = 0

        # Calculate average MAB reward of current neighbors
        for neighbor_id in self.neighbors:
            if neighbor_id in self.mab_arms:
                total_mab_reward += self.mab_arms[neighbor_id].average_reward
                num_mab_arms += 1

        avg_mab_reward = total_mab_reward / num_mab_arms if num_mab_arms > 0 else 0.0

        # Discretize the average MAB reward into 4 bins
        # Assuming avg_mab_reward is normalized or capped, so multiplying by 4 gives 0-3 range
        discretized_avg_mab_reward = int(avg_mab_reward * 4)
        # Ensure it's within expected range [0, 3] if avg_mab_reward is normalized to [0,1]
        discretized_avg_mab_reward = max(0, min(3, discretized_avg_mab_reward))

        return (is_dest_neighbor, discretized_avg_mab_reward)

    def _select_sarsa_action(self, state, current_neighbors_ids, packet=None, simulation_environment_ref=None):
        """Selects a forwarder using an epsilon-greedy policy based on the SARSA Q-table.
        If multiple actions tie on Q-value, prefer the neighbor closer to the packet's destination.
        """
        if not current_neighbors_ids:
            return None # No neighbors to forward to

        if random.random() < self.sarsa_epsilon: # Explore
            return random.choice(current_neighbors_ids)
        else: # Exploit
            if state not in self.q_table:
                self.q_table[state] = {}

            # Initialize Q-values for new (state, action) pairs if not present
            for neighbor_id in current_neighbors_ids:
                if neighbor_id not in self.q_table[state]:
                    self.q_table[state][neighbor_id] = 0.0

            # Select action with maximum Q-value
            max_q_value = -float('inf')
            best_actions = []

            for neighbor_id in current_neighbors_ids:
                q_value = self.q_table[state].get(neighbor_id, 0.0) # Use .get with default 0.0 for safety
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_actions = [neighbor_id]
                elif q_value == max_q_value:
                    best_actions.append(neighbor_id)

            if len(best_actions) <= 1 or packet is None or simulation_environment_ref is None:
                return random.choice(best_actions)

            # Prefer neighbor closer to destination as tie-breaker
            dest_node = simulation_environment_ref.get_node_by_address(packet.destination_ip)
            if not dest_node:
                return random.choice(best_actions)
            def distance_to_dest(nid):
                n = simulation_environment_ref.get_node_by_id(nid)
                return simulation_environment_ref._calculate_euclidean_distance(n, dest_node) if n else float('inf')
            best_actions.sort(key=distance_to_dest)
            return best_actions[0]

    def _update_sarsa_q_table(self, prev_state, prev_action, reward, next_state, next_action):
        """Performs the SARSA update rule."""
        if prev_state is None or prev_action is None:
            return # No previous state-action to update

        # Ensure prev_state and prev_action exist in Q-table
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {}
        if prev_action not in self.q_table[prev_state]:
            self.q_table[prev_state][prev_action] = 0.0

        current_q = self.q_table[prev_state][prev_action]
        next_q = 0.0

        if next_state is not None and next_action is not None:
            # Ensure next_state and next_action exist in Q-table for calculation
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
            if next_action not in self.q_table[next_state]:
                self.q_table[next_state][next_action] = 0.0
            next_q = self.q_table[next_state][next_action]

        # SARSA update rule
        self.q_table[prev_state][prev_action] = current_q + self.sarsa_alpha * (
            reward + self.sarsa_gamma * next_q - current_q
        )

    def select_forwarder_mab(self, epsilon):
        """Selects a forwarder using the MAB (epsilon-greedy) strategy."""
        # This method is becoming deprecated as SARSA takes over forwarding decisions
        # but kept for compatibility if needed elsewhere or for comparison.
        if not self.mab_arms or not self.neighbors:
            return None # No known potential forwarders

        # Filter MAB arms to only consider current neighbors
        current_neighbor_arms = {n_id: self.mab_arms[n_id] for n_id in self.neighbors if n_id in self.mab_arms}

        if not current_neighbor_arms:
            return None

        if random.random() < epsilon: # Explore
            return random.choice(list(current_neighbor_arms.keys()))
        else: # Exploit
            best_arm_ids = []
            max_avg_reward = -1.0 # Assuming rewards are non-negative
            for node_id, arm in current_neighbor_arms.items():
                if arm.average_reward > max_avg_reward:
                    max_avg_reward = arm.average_reward
                    best_arm_ids = [node_id]
                elif arm.average_reward == max_avg_reward:
                    best_arm_ids.append(node_id)
            return random.choice(best_arm_ids) # Break ties randomly

    def update_mab_reward(self, forwarder_node_id, reward):
        """Updates the reward for a selected MAB arm."""
        # This method is for MAB, not directly used by SARSA's core update,
        # but average MAB reward is used in SARSA state representation.
        if forwarder_node_id in self.mab_arms:
            arm = self.mab_arms[forwarder_node_id]
            arm.plays += 1
            arm.rewards += reward
        else:
            print(f"Warning: Node {self.node_id} tried to update MAB for unknown forwarder {forwarder_node_id}")

    def send_data_packet(self, destination_node_address, simulation_environment_ref, ttl=5, payload=None):
        """Creates an OppDataPacket and adds it to its buffer."""
        self.last_packet_id += 1
        packet = OppDataPacket(
            source_ip=self.address,
            destination_ip=destination_node_address,
            ttl=ttl,
            payload=payload,
            packet_id=f"{self.node_id}-{self.last_packet_id}"
        )
        self.packet_buffer.append(packet)

    def process_packet(self, packet, sender_node_id, simulation_environment_ref):
        """Handles an incoming packet."""
        # If the packet is for this node, deliver it
        if packet.destination_ip == self.address:
            simulation_environment_ref.delivered_packets.append(packet)
            # Reward the sender for successful delivery
            if sender_node_id is not None: # Not None if this is a forwarded packet, not initial transmission
                sender_node = simulation_environment_ref.get_node_by_id(sender_node_id)
                if sender_node:
                    # Perform SARSA update for the sender node for successful delivery
                    # The sender node will have stored the prev_state and prev_action on the packet
                    reward = 1.0 # Reward for successful delivery to final destination
                    # The terminal state is indicated by None for next_state and next_action
                    sender_node._update_sarsa_q_table(
                        packet.sarsa_prev_state, packet.sarsa_prev_action, reward, None, None
                    )
        else:
            # If not for this node, add to buffer for forwarding
            # Check if this node already has this packet (to prevent loops/duplicates)
            if packet.packet_id not in [p.packet_id for p in self.packet_buffer]:
                self.packet_buffer.append(packet)

    def forward_packet(self, simulation_environment_ref):
        """Selects a packet from buffer and attempts to forward it using SARSA."""
        if not self.packet_buffer:
            return

        # For simplicity, pick the first packet in the buffer
        packet_to_forward = self.packet_buffer[0]

        # Decrease TTL. If TTL <= 0, drop the packet.
        packet_to_forward.ttl -= 1
        if packet_to_forward.ttl <= 0:
            self.packet_buffer.pop(0) # Remove from buffer
            # If packet is dropped due to TTL expiry, it's a negative reward for the action that led to it.
            # This action is stored as sarsa_prev_state/action on the packet.
            if packet_to_forward.sarsa_prev_state is not None and packet_to_forward.sarsa_prev_action is not None:
                # The node currently holding the packet (self) is responsible for the update
                # for the action it took previously on this packet, leading to its expiry.
                self._update_sarsa_q_table(
                    packet_to_forward.sarsa_prev_state, packet_to_forward.sarsa_prev_action, -1.0, None, None
                )
            return # Do not attempt to forward

        # Obtain the current SARSA state (S)
        state_s = self._get_sarsa_state(packet_to_forward, simulation_environment_ref)

        # Identify all valid current neighbors as potential actions
        current_neighbors_ids = list(self.neighbors.keys())

        # Select the action (chosen forwarder's ID) using SARSA's epsilon-greedy policy (A)
        action_a = self._select_sarsa_action(state_s, current_neighbors_ids, packet_to_forward, simulation_environment_ref)

        if action_a is None:
            # No forwarder available or chosen, packet remains in buffer
            # In a real SARSA setting, this might also be a terminal state or a self-loop with negative reward
            # For simplicity, we just return and keep the packet in buffer to retry
            return

        # Store current state and chosen action on the packet.
        # These will be the (prev_state, prev_action) for the *next* update related to this packet,
        # which will be initiated by `handle_forwarding_attempt` for *this* node (self)
        # or by `process_packet` if it reaches its destination.
        packet_to_forward.sarsa_prev_state = state_s
        packet_to_forward.sarsa_prev_action = action_a

        # Simulate sending to the chosen forwarder via the environment
        success = simulation_environment_ref.handle_forwarding_attempt(self, packet_to_forward, action_a)

        if success:
            # If successfully forwarded, remove from buffer
            self.packet_buffer.pop(0)
        # Note: SARSA update for this node (self) is now handled in SimulationEnvironment.handle_forwarding_attempt

    def __repr__(self):
        public_key_display = self._get_public_key_pem()[:20] + "..." if self.public_key else "N/A"
        mab_arms_info = f"{len(self.mab_arms)} arms"
        cert_display = "N/A" if not self.node_certificate else (self.node_certificate['public_key_pem'][:20] + "..." if 'public_key_pem' in self.node_certificate else "Valid")
        return (
            f"Node(ID: {self.node_id}, Addr: {self.address}, MAC: {self.mac_address}, "
            f"Pos: ({self.x:.2f},{self.y:.2f},{self.z:.2f}), Speed: {self.speed:.2f}, InstSpeed: {self.instantaneous_speed:.2f}, "
            f"Dest: ({self.destination_x:.2f},{self.destination_y:.2f},{self.destination_z:.2f})" if self.destination_x is not None else "Dest: N/A, "
            f"Neighbors: {len(self.neighbors)}, MAB: {mab_arms_info}, Packets: {len(self.packet_buffer)}, Cert: {cert_display})"
        )

print("Node class updated successfully for SARSA packet forwarding and TTL expiry updates.")

# Modified HelloPacket class (updated for secure data and sender certificate, and instantaneous speed)
class HelloPacket:
    def __init__(self, sender_node_id, sender_public_key_pem, encrypted_payload, nonce, tag, sender_node_certificate, instantaneous_speed): # instantaneous_speed is now a required parameter
        self.sender_node_id = sender_node_id
        self.sender_public_key_pem = sender_public_key_pem
        self.encrypted_payload = encrypted_payload
        self.nonce = nonce
        self.tag = tag
        self.sender_node_certificate = sender_node_certificate # New: Store the sender's node certificate
        self.instantaneous_speed = instantaneous_speed # Store instantaneous speed

    def __repr__(self):
        cert_display = "N/A" if not self.sender_node_certificate else (self.sender_node_certificate['public_key_pem'][:20] + "..." if 'public_key_pem' in self.sender_node_certificate else "Valid")
        return (
            f"HelloPacket(SenderID: {self.sender_node_id}, "
            f"SenderPK: {self.sender_public_key_pem[:20]}..., "
            f"EncryptedPayloadLen: {len(self.encrypted_payload)}, "
            f"Nonce: {self.nonce.hex()[:8]}..., "
            f"Tag: {self.tag.hex()[:8]}..., "
            f"Cert: {cert_display}, InstSpeed: {self.instantaneous_speed:.2f})") # Added InstSpeed, now always available

print("HelloPacket class updated successfully for secure data, sender certificate, and instantaneous speed.")

# Modified Configuration class (updated for Root CA key generation and Node certificate signing)
class Configuration:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

        # Generate actual ECCDSA private and public keys for the Root CA
        self.root_ca_private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self.root_ca_public_key = self.root_ca_private_key.public_key()
        self.root_ca_public_key_pem = self._get_public_key_pem(self.root_ca_public_key)

        self.nodes = []
        self._create_nodes()

    def _get_public_key_pem(self, public_key):
        """Helper method to return a public key in PEM format."""
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def _create_nodes(self):
        # print(f"Creating {self.num_nodes} nodes...")
        for i in range(self.num_nodes):
            node_id = i + 1
            address = f"192.168.1.{node_id}"
            mac_address = f"00:00:00:00:00:{node_id:02x}"
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            z = random.uniform(0, 1000)
            speed = random.uniform(1, 10)

            # Pass the Root CA's public key PEM string to each Node
            node = Node(node_id, address, mac_address, x, y, z, speed, self.root_ca_public_key_pem)

            # --- Node Certificate Generation and Signing ---
            # 1. Construct data to be signed for the node's certificate
            cert_data_content = {
                "node_id": node.node_id,
                "mac_address": node.mac_address,
                "public_key_pem": node._get_public_key_pem() # Node's public key
            }
            cert_data_bytes = json.dumps(cert_data_content, sort_keys=True).encode('utf-8')

            # 2. Use the Root CA's private key to sign these serialized certificate data bytes
            signature = self.root_ca_private_key.sign(
                cert_data_bytes,
                ec.ECDSA(hashes.SHA256())
            )

            # 3. Create the node_certificate dictionary
            node_certificate = {
                "node_id": node.node_id,
                "mac_address": node.mac_address,
                "public_key_pem": node._get_public_key_pem(),
                "signature": base64.b64encode(signature).decode('utf-8') # Base64 encode signature for JSON compatibility
            }

            # 4. Assign this generated node_certificate to the node
            node.node_certificate = node_certificate
            # --- End Node Certificate Generation ---

            self.nodes.append(node)
        # print(f"Successfully created {len(self.nodes)} nodes.")

print("Configuration class updated successfully for Root CA key generation and Node certificate signing.")

# Modified SimulationEnvironment class (updated for neighbor discovery, MAB arm initialization, packet forwarding, and get_node_by_address)
class SimulationEnvironment:
    COMMUNICATION_RANGE = 250 # meters (per requirement)

    def __init__(self, num_nodes):
        print("Initializing SimulationEnvironment...")
        config = Configuration(num_nodes)
        self.nodes = config.nodes
        self.x_bounds = (0, 1000)
        self.y_bounds = (0, 1000)
        self.z_bounds = (0, 1000)
        self.current_time = 0.0
        self.node_map = {node.node_id: node for node in self.nodes} # New: For quick lookup by node_id
        self.node_address_map = {node.address: node for node in self.nodes} # New: For quick lookup by address
        self.delivered_packets = [] # New: To track successfully delivered packets
        self.config = config # Store the Configuration object

        self.NETWORK_ASCON_KEY = random.getrandbits(128).to_bytes(16, 'big') # 128-bit key for Ascon

        for node in self.nodes:
            node.symmetric_key = self.NETWORK_ASCON_KEY

        print(f"SimulationEnvironment initialized with {len(self.nodes)} nodes within bounds: X {self.x_bounds}, Y {self.y_bounds}, Z {self.z_bounds}.")
        print(f"Network-wide ASCON symmetric key generated: {self.NETWORK_ASCON_KEY.hex()[:10]}...")
        print(f"Initial positions and security/MAB details of first 3 nodes:")
        for i in range(min(3, len(self.nodes))):
            print(self.nodes[i])

    def _calculate_euclidean_distance(self, node1, node2):
        """Calculates the 3D Euclidean distance between two nodes."""
        return math.sqrt(
            (node1.x - node2.x)**2 +
            (node1.y - node2.y)**2 +
            (node1.z - node2.z)**2
        )

    def get_node_by_id(self, node_id):
        """Returns the Node object corresponding to the given node_id."""
        return self.node_map.get(node_id)

    def get_node_by_address(self, node_address):
        """Returns the Node object corresponding to the given node_address."""
        return self.node_address_map.get(node_address)

    def _perform_neighbor_discovery_and_exchange_hellos(self):
        """Performs neighbor discovery and simulates Secure HelloPacket exchange.
        Logic:
        - Clear neighbor lists each cycle.
        - If current Euclidean distance <= COMMUNICATION_RANGE, attempt secure Hello exchange.
        - Accept neighbor only if certificate verifies, payload signature verifies,
          decryption succeeds, and mobility heuristic predicts neighbor will remain within range.
        """
        for node in self.nodes:
            node.neighbors = {}

        num_nodes = len(self.nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node1 = self.nodes[i]
                node2 = self.nodes[j]

                distance = self._calculate_euclidean_distance(node1, node2)

                if distance <= self.COMMUNICATION_RANGE:
                    # Initialize MAB arms for potential neighbors (used in SARSA state)
                    if node2.node_id not in node1.mab_arms:
                        node1.mab_arms[node2.node_id] = MABArm()
                    if node1.node_id not in node2.mab_arms:
                        node2.mab_arms[node1.node_id] = MABArm()

                    # Secure Hello exchange in both directions
                    hello_packet_from_node1 = node1.create_hello_packet()
                    self._process_incoming_hello_packet(node2, hello_packet_from_node1)

                    hello_packet_from_node2 = node2.create_hello_packet()
                    self._process_incoming_hello_packet(node1, hello_packet_from_node2)

    def _process_incoming_hello_packet(self, receiver_node, hello_packet):
        """Processes an incoming secure HelloPacket, performing decryption and verification.
        Adds sender to `receiver_node.neighbors` only if:
        - Certificate signature verifies with Root CA public key
        - Payload decrypts and signature verifies with sender public key
        - Mobility heuristic predicts sender remains within range soon
        """
        try:
            sender_certificate = hello_packet.sender_node_certificate
            if not sender_certificate:
                return

            cert_node_id = sender_certificate['node_id']
            cert_mac_address = sender_certificate['mac_address']
            cert_public_key_pem = sender_certificate['public_key_pem']
            cert_signature_b64 = sender_certificate['signature']

            reconstructed_cert_data_content = {
                "node_id": cert_node_id,
                "mac_address": cert_mac_address,
                "public_key_pem": cert_public_key_pem
            }
            reconstructed_cert_data_bytes = json.dumps(reconstructed_cert_data_content, sort_keys=True).encode('utf-8')
            decoded_cert_signature = base64.b64decode(cert_signature_b64)

            dummy_node = Node(0, "0.0.0.0", "00:00:00:00:00:00", 0, 0, 0, 0, self.config.root_ca_public_key_pem)

            is_cert_signature_valid = dummy_node.verify_signature(
                self.config.root_ca_public_key_pem,
                reconstructed_cert_data_bytes,
                decoded_cert_signature
            )

            if not is_cert_signature_valid:
                return

            decrypted_signed_payload = receiver_node.decrypt_data_ascon(
                hello_packet.encrypted_payload, hello_packet.nonce, hello_packet.tag
            )

            sig_len = int.from_bytes(decrypted_signed_payload[:4], 'big')
            signature = decrypted_signed_payload[4 : 4 + sig_len]
            hello_bytes = decrypted_signed_payload[4 + sig_len :]

            is_hello_signature_valid = receiver_node.verify_signature(
                cert_public_key_pem,
                hello_bytes, signature
            )

            if not is_hello_signature_valid:
                return

            hello_content = json.loads(hello_bytes.decode('utf-8'))
            sender_node = self.get_node_by_id(hello_content['node_id'])
            if not sender_node:
                return

            # Mobility-based decision: keep only if predicted to remain in range
            inst_speed = float(hello_content.get("instantaneous_speed", 0.0))
            cur_distance = self._calculate_euclidean_distance(receiver_node, sender_node)
            if self._should_keep_neighbor(cur_distance, inst_speed):
                receiver_node.neighbors[sender_node.node_id] = sender_node
        except Exception:
            # Drop malformed/invalid hello silently
            pass

    def _should_keep_neighbor(self, current_distance, neighbor_inst_speed):
        """Heuristic: accept neighbor if currently in range or predicted drift in next
        time unit keeps it within range. Prediction window tau=1.0.
        We lack direction; approximate by allowing margin = speed * tau.
        """
        tau = 1.0
        margin = max(0.0, neighbor_inst_speed * tau)
        return (current_distance <= self.COMMUNICATION_RANGE) or (
            current_distance <= self.COMMUNICATION_RANGE + margin
        )

    def simulate_packet_transmission(self, source_node_id, destination_node_id, ttl=5, payload=None):
        """Initiates packet sending from a source node to a destination node's address."""
        source_node = self.get_node_by_id(source_node_id)
        destination_node = self.get_node_by_id(destination_node_id)

        if not source_node:
            print(f"Error: Source node with ID {source_node_id} not found.")
            return
        if not destination_node:
            print(f"Error: Destination node with ID {destination_node_id} not found.")
            return

        source_node.send_data_packet(destination_node.address, self, ttl=ttl, payload=payload)
        print(f"Initial packet {source_node.packet_buffer[-1].packet_id} sent by Node {source_node_id} for Node {destination_node_id}.")

    def handle_forwarding_attempt(self, sending_node, packet, chosen_forwarder_id):
        """Handles the actual transfer of a packet from sending_node to chosen_forwarder_id
           and performs SARSA update for the sending_node."""
        chosen_forwarder_node = self.get_node_by_id(chosen_forwarder_id)

        success = False
        reward = 0.0
        next_state_s_prime = None
        next_action_a_prime = None

        if chosen_forwarder_node and chosen_forwarder_id in sending_node.neighbors:
            success = True
            reward = 0.1 # Intermediate reward for successful relay
            # Simulate successful transfer by calling process_packet on the receiver
            # The process_packet might deliver the packet to its final destination
            chosen_forwarder_node.process_packet(packet, sending_node.node_id, self)

            # After processing by chosen_forwarder_node, if it was not the final destination,
            # we predict the next action for SARSA update for the sending_node.
            if packet.destination_ip != chosen_forwarder_node.address:
                # Predict the next state (S') and action (A') from the chosen_forwarder_node's perspective
                next_state_s_prime = chosen_forwarder_node._get_sarsa_state(packet, self)
                next_action_a_prime = chosen_forwarder_node._select_sarsa_action(next_state_s_prime, list(chosen_forwarder_node.neighbors.keys()))
            # Else, if chosen_forwarder_node was the destination, it's a terminal state for SARSA of the sending_node,
            # so next_state_s_prime and next_action_a_prime remain None.

        else:
            # Forwarding failed (e.g., node moved out of range or no longer a valid neighbor)
            reward = -0.5 # Penalty for failed relay
            # If forwarding failed, the packet remains in the sending_node's buffer.
            # No next state/action (S', A') because the attempted action failed.

        # Perform SARSA update for the sending_node
        # The (prev_state, prev_action) for the sending_node are stored on the packet itself
        sending_node._update_sarsa_q_table(
            packet.sarsa_prev_state, packet.sarsa_prev_action, reward, next_state_s_prime, next_action_a_prime
        )

        # Update MAB rewards to inform state discretization
        sending_node.update_mab_reward(chosen_forwarder_id, reward)

        return success

    def simulate_time_step(self, time_step):
        self.current_time += time_step

        # 1. Update positions for all nodes
        for node in self.nodes:
            node.update_position(time_step, self.current_time, self.x_bounds, self.y_bounds, self.z_bounds) # Pass current_time

        # 2. Perform neighbor discovery and exchange Secure Hello messages
        self._perform_neighbor_discovery_and_exchange_hellos()

        # 3. Iterate through nodes and allow them to forward packets
        for node in self.nodes:
            node.forward_packet(self)


print("SimulationEnvironment class updated successfully for SARSA integration in handle_forwarding_attempt.")
print("Node.process_packet and Node.forward_packet (TTL expiry) logic checked and refined for SARSA updates.")

# ---- Simple driver to run a simulation and report Packet Delivery Ratio (PDR) ----
if __name__ == "__main__":
    # Parameters
    NUM_NODES = 100
    SOURCE_ID = 1
    DEST_ID = 2
    NUM_PACKETS = 4
    MAX_STEPS = 200
    TIME_STEP = 1.0

    env = SimulationEnvironment(NUM_NODES)

    # Send initial packets from source to destination
    for _ in range(NUM_PACKETS):
        env.simulate_packet_transmission(SOURCE_ID, DEST_ID, ttl=8, payload=None)

    # Step the simulation
    for step in range(MAX_STEPS):
        env.simulate_time_step(TIME_STEP)
        # Stop early if all packets delivered
        if len(env.delivered_packets) >= NUM_PACKETS:
            break

    delivered = len([p for p in env.delivered_packets if str(SOURCE_ID) in str(p.packet_id)])
    total = NUM_PACKETS
    pdr = delivered / total if total > 0 else 0.0

    print(f"Simulation finished in {step+1} steps.")
    # Quick neighbor stats for first 5 nodes
    for i in range(1, 6):
        node = env.get_node_by_id(i)
        if node:
            print(f"Node {i}: neighbors={len(node.neighbors)} inst_speed={node.instantaneous_speed:.2f}")
    print(f"Packets delivered: {delivered}/{total} | PDR = {pdr:.2f}")