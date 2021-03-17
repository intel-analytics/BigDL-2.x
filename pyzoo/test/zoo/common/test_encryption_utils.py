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

import pytest
import string
import random

from zoo.common.encryption_utils import *


class TestEncryption(object):

    def test_aes128_cbc_bytes(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_bytes = encrypt_bytes_with_AES_CBC(random_str.encode(), 'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes, 'analytics-zoo', 'intel-analytics')
        assert dec_bytes == random_str.encode(), \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes256_cbc_bytes(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        enc_bytes = encrypt_bytes_with_AES_CBC(random_str.encode("utf-8"), 'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == random_str.encode("utf-8"), \
            "Check AES CBC 256 encryption and decryption result"

    def test_aes128_gcm_bytes(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_bytes = encrypt_bytes_with_AES_GCM(random_str.encode(), 'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes, 'analytics-zoo', 'intel-analytics')
        assert dec_bytes == random_str.encode(), \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes256_gcm_bytes(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_bytes = encrypt_bytes_with_AES_GCM(random_str.encode(), 'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == random_str.encode(), \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes128_cbc(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_str = encrypt_with_AES_CBC(random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == random_str, \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes256_cbc(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_str = encrypt_with_AES_CBC(random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == random_str, \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes128_gcm(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_str = encrypt_with_AES_GCM(random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == random_str, \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes256_gcm(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_str = encrypt_with_AES_GCM(random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == random_str, \
            "Check AES GCM 128 encryption and decryption result"



if __name__ == "__main__":
    pytest.main([__file__])
