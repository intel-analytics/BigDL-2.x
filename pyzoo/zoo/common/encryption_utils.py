import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import base64
import hashlib


def get_private_key(secret_key, salt, key_len=128):
    """
    Generate AES required random secret/privacy key
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :return: random key
    """
    # AES key_len means key bit length, not string/char length
    # 16 for 128 and 32 for 256
    bit_len = (key_len / 128) * 16
    return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, bit_len)


def encrypt_with_AES_CBC(plain_text, secret_key, salt, key_len=128, block_size=16):
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    # Align with Java AES/CBC/PKCS5PADDING
    padder = padding.PKCS7(key_len).padder()
    data = padder.update(plain_text)
    # create Cipher
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(data) + encryptor.finalize()
    return iv + ct


def decrypt_with_AES_CBC(cipher_text, secret_key, salt, key_len=128, block_size=16):
    key = get_private_key(secret_key, salt, key_len)
    iv = cipher_text[:block_size]
    unpadder = padding.PKCS7(key_len).unpadder()
    ct = unpadder.update(cipher_text[block_size:])
    # create Cipher
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    return decryptor.update(ct) + decryptor.finalize()
