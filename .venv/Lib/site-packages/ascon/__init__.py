from ascon._ascon import ascon_encrypt as encrypt
from ascon._ascon import ascon_decrypt as decrypt
from ascon._ascon import ascon_hash as hash
from ascon._ascon import ascon_mac as mac

__all__ = ["encrypt", "decrypt", "hash", "mac"]
