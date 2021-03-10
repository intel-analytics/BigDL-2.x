#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import hashlib

from Crypto import Random
from Crypto.Cipher import AES

block_size = 16
pad = lambda s: s + (block_size - len(s) % block_size) * chr(block_size - len(s) % block_size)
unpad = lambda s: s[0:-ord(s[-1:])]


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
    key_len = (key_len / 128) * 16
    return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, key_len)


def encrypt_with_AES(message, secret_key, salt, key_len=128):
    """
    Using AES to encrypt string
    :param message: plain test in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :return: AES encrypted string
    """
    private_key = get_private_key(secret_key, salt, key_len)
    iv = Random.new().read(AES.block_size)
    message = pad(message)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    cipher_bytes = base64.b64encode(iv + cipher.encrypt(message))
    return bytes.decode(cipher_bytes)


def decrypt_with_AES(encoded, secret_key, salt, key_len=128):
    """
    Decrypted AES encrypted string
    :param encoded: AES encrypted text in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :return: plain text in string
    """
    private_key = get_private_key(secret_key, salt, key_len)
    cipher_text = base64.b64decode(encoded)
    iv = cipher_text[:AES.block_size]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    plain_bytes = unpad(cipher.decrypt(cipher_text[block_size:]))
    return bytes.decode(plain_bytes)
