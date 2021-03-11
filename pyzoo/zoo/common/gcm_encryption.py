import base64
import hashlib

from Crypto import Random
from Crypto.Cipher import AES

block_size = 12


def get_private_key(secret_key, salt, key_len=128):
    key_len = (key_len / 128) * 16
    return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, key_len)


def encrypt_with_AES_GCM(message, secret_key, salt, key_len=128):
    private_key = get_private_key(secret_key, salt, key_len)
    iv = Random.new().read(block_size)
    cipher = AES.new(private_key, AES.MODE_GCM, nonce=iv)
    # pay attention to tag
    cipher_text, tag = cipher.encrypt_and_digest(message)
    cipher_bytes = base64.b64encode(iv + cipher_text + tag)
    return bytes.decode(cipher_bytes)


def decrypt_with_AES_GCM(encoded, secret_key, salt, key_len=128):
    private_key = get_private_key(secret_key, salt, key_len)
    cipher_text = base64.b64decode(encoded)
    iv = cipher_text[:block_size]
    cipher = AES.new(private_key, AES.MODE_GCM, nonce=iv)
    # Default tag size is 16, i.e., 128 bits
    plain_bytes = cipher.decrypt(cipher_text[block_size:-16])
    return bytes.decode(plain_bytes)
