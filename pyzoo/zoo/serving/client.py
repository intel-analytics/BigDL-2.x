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
import cv2
import yaml
import redis
import datetime
import time


class API:
    """
    base level of API control
    select data pipeline here, Redis/Kafka/...
    interface preserved for API class
    """
    def __init__(self, host, port):

        # try:
        #     file_path = "config.yaml"
        #
        #     with open(file_path) as f:
        #         config = yaml.load(f)
        #         if not config['data']['src']:
        #             host_port = ["localhost", "6379"]
        #         else:
        #             host_port = config['data']['src'].split(":")
        #         config['data']['host'] = host_port[0]
        #         config['data']['port'] = host_port[1]
        # except Exception as e:
        #     print("config file does not exist in your current directory,"
        #           "will use default value")
        #
        #
        # self.db = redis.StrictRedis(host=config['data']['host'],
        #                             port=config['data']['port'], db=0)
        self.host = host
        self.port = port

        self.db = redis.StrictRedis(host=self.host, port=self.port, db=0)

        try:
            self.db.xgroup_create("image_stream", "serving")
        except Exception:
            print("redis group exist, will not create new one")


class InputQueue(API):
    def __init__(self, host="localhost", port="6379", data_shape="3,224,224"):
        super().__init__(host, port)
        self.c, self.h, self.w = None, None, None
        self.data_shape = data_shape.split(",")
        # TODO: these params can be read from config in future
        self.input_threshold = 0.6
        self.interval_if_error = 1
        self.data_shape_check()

    def data_shape_check(self):
        for num in self.data_shape:
            num = int(num)
            if num <= 0:
                raise Exception("Your image shape config is invalid, "
                                "your config shape is" + str(self.data_shape)
                                + "no negative value is allowed.")
            if 0 < num < 5:
                self.c = num
                continue
            if not self.h:
                self.h = num
            else:
                self.w = num
        return

    def enqueue_image(self, uri, img):
        """
        :param id: String you use to identify this record
        :param data: Data, ndarray type
        :return:
        """
        if isinstance(img, str):
            img = cv2.imread(str)
            if not img:
                print("You have pushed an image with path: ",
                      img, "the path is invalid, skipped.")
                return

        # force resize here to avoid input image shape inconsistent
        # if the shape is consistent, it would not affect the data
        img = cv2.resize(img, (self.h, self.w))
        data = cv2.imencode(".jpg", img)[1]

        img_encoded = self.base64_encode_image(data)

        d = {"uri": uri, "image": img_encoded}

        inf = self.db.info()
        # try:
        #     inf['maxmemory']
        # except Exception as e:
        #     self.db.xadd("image_stream", d)
        #     return

        try:
            if inf['used_memory'] >= inf['maxmemory'] * self.input_threshold:
                raise redis.exceptions.ConnectionError
            self.db.xadd("image_stream", d)
            print("Write to Redis successful")
        except redis.exceptions.ConnectionError:
            print("Redis queue is full, please wait for inference "
                  "or delete the unprocessed records.")
            time.sleep(self.interval_if_error)

        except redis.exceptions.ResponseError as e:
            print(e, "Redis memory is full, please dequeue or delete.")
            time.sleep(self.interval_if_error)

    @staticmethod
    def base64_encode_image(img):
        # base64 encode the input NumPy array
        return base64.b64encode(img).decode("utf-8")


class OutputQueue(API):
    def __init__(self, host="localhost", port="6379"):
        super().__init__(host, port)

    def dequeue(self):
        res_list = self.db.keys('result:*')
        decoded = {}
        for res in res_list:
            res_dict = self.db.hgetall(res.decode('utf-8'))
            res_id = res.decode('utf-8').split(":")[1]
            res_value = res_dict[b'value'].decode('utf-8')
            decoded[res_id] = res_value
            self.db.delete(res)
        return decoded

    def query(self, uri):
        res_dict = self.db.hgetall("result:"+uri)

        if not res_dict or len(res_dict) == 0:
            return "{}"
        return res_dict[b'value'].decode('utf-8')
