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

from zoo.common.encryption import encrypt_with_AES, decrypt_with_AES


class TestEncryption(object):

    def test_aes128(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        # random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        enc_str = encrypt_with_AES(random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == random_str, \
            "Check AES128 encryption and decryption result"

    def test_aes256(self):
        letters = string.ascii_lowercase
        random_str = ''.join(random.choice(letters) for i in range(100))
        enc_str = encrypt_with_AES(random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == random_str, \
            "Check AES256 encryption and decryption result"


if __name__ == "__main__":
    pytest.main([__file__])
