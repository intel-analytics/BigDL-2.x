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

import numpy as np
import base64
import cv2
import yaml
import redis


class RedisQueue:
    def __init__(self, file_path=None):
        if file_path:
            with open(file_path) as f:
                config = yaml.load(f)
                host_port = config['data']['src'].split(":")
                config['data']['host'] = host_port[0]
                config['data']['port'] = host_port[1]
        else:
            config = {'data': {'host': "localhost", 'port': "6379", 'shape': "3,224,224"}}

        self.db = redis.StrictRedis(host=config['data']['host'],
                                    port=config['data']['port'], db=0)

        self.data_shape = config['data']['shape'].split(",")
        for i in range(len(self.data_shape)):
            self.data_shape[i] = int(self.data_shape[i])

    def enqueue_image(self, uri, img):
        """
        :param id: String you use to identify this record
        :param data: Data, ndarray type
        :return:
        """
        if img.shape[0] != self.data_shape[1] or img.shape[1] != self.data_shape[2]:

            raise AssertionError("Your image shape " + str(img.shape) + " does not match "
                                 "that in your config " + str(self.data_shape) + ", please check")
        data = cv2.imencode(".jpg", img)[1]

        img_encoded = self.base64_encode_image(data)
        d = {"uri": uri, "image": img_encoded}
        self.db.xadd("image_stream", d)

    # def get_results(self, key):
    #     return self.db.hgetall(key)
    def get_results(self):
        res_list = self.db.keys('result:*')
        decoded = {}
        for res in res_list:
            res_dict = self.db.hgetall(res.decode('utf-8'))
            res_id = res_dict[b'_1'].decode('utf-8')
            res_value = res_dict[b'_2'].decode('utf-8')
            decoded[res_id] = res_value
        return decoded

    @staticmethod
    def base64_encode_image(img):
        # base64 encode the input NumPy array
        return base64.b64encode(img).decode("utf-8")
