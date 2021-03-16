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
    ct_bytes = encrypt_bytes_with_AES_CBC(plain_text.encode(),
                                          secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()


def decrypt_with_AES_CBC(cipher_text, secret_key, salt, key_len=128, block_size=16):
    plain_bytes = decrypt_bytes_with_AES_CBC(base64.b64decode(cipher_text),
                                             secret_key, salt, key_len, block_size)
    return plain_bytes.decode()


def encrypt_bytes_with_AES_CBC(plain_text_bytes, secret_key, salt, key_len=128, block_size=16):
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    # Align with Java AES/CBC/PKCS5PADDING
    padder = padding.PKCS7(key_len).padder()
    data = padder.update(plain_text_bytes)
    data += padder.finalize()
    # create Cipher
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ct = encryptor.update(data) + encryptor.finalize()
    return iv + ct


def decrypt_bytes_with_AES_CBC(cipher_text_bytes, secret_key, salt, key_len=128, block_size=16):
    key = get_private_key(secret_key, salt, key_len)
    iv = cipher_text_bytes[:block_size]
    # create Cipher
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    ct = decryptor.update(cipher_text_bytes[block_size:]) + decryptor.finalize()
    # unpadding
    unpadder = padding.PKCS7(key_len).unpadder()
    ct = unpadder.update(ct)
    ct += unpadder.finalize()
    return ct


def encrypt_with_AES_GCM(plain_text, secret_key, salt, key_len=128, block_size=12):
    ct_bytes = encrypt_bytes_with_AES_GCM(plain_text.encode(),
                                          secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()


def decrypt_with_AES_GCM(cipher_text, secret_key, salt, key_len=128, block_size=12):
    plain_bytes = decrypt_bytes_with_AES_GCM(base64.b64decode(cipher_text),
                                             secret_key, salt, key_len, block_size)
    return plain_bytes.decode()


def encrypt_bytes_with_AES_GCM(plain_text_bytes, secret_key, salt, key_len=128, block_size=12):
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    # create Cipher
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
    ct = encryptor.update(plain_text_bytes) + encryptor.finalize()
    return iv + ct + encryptor.tag


def decrypt_bytes_with_AES_GCM(cipher_text_bytes, secret_key, salt, key_len=128, block_size=12):
    key = get_private_key(secret_key, salt, key_len)
    tag = cipher_text_bytes[-16:]
    iv = cipher_text_bytes[:block_size]
    # create Cipher
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
    # 16 for tag
    ct = decryptor.update(cipher_text_bytes[block_size:-16]) + decryptor.finalize()
    return ct
