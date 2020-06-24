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
import time
import numpy as np
import pyarrow as pa
from zoo.serving.schema import *


class API:
    """
    base level of API control
    select data pipeline here, Redis/Kafka/...
    interface preserved for API class
    """
    def __init__(self):

        try:
            file_path = "config.yaml"
        except Exception:
            raise EOFError("config file does not exist. Please check your config"
                           "at analytics-zoo/docker/cluster-serving/config.yaml")

        try:
            with open(file_path) as f:
                config = yaml.load(f)
                if not config['data']['src']:
                    host_port = ["localhost", "6379"]
                else:
                    host_port = config['data']['src'].split(":")
                config['data']['host'] = host_port[0]
                config['data']['port'] = host_port[1]
        except Exception:
            config = {}
            config['data'] = {}
            config['data']['host'], config['data']['port'] = "localhost", "6379"
            config['data']['image_shape'] = None

        self.db = redis.StrictRedis(host=config['data']['host'],
                                    port=config['data']['port'], db=0)
        # self.db = redis.StrictRedis(host="10.239.47.210",
        #                             port="16380", db=0)
        try:
            self.db.xgroup_create("image_stream", "serving")
            self.db.xgroup_create("tensor_stream", "serving")
        except Exception:
            print("redis group exist, will not create new one")

        if not config['data']['image_shape']:
            self.data_shape = ["3", "224", "224"]
        else:
            self.data_shape = config['data']['image_shape'].split(",")
        for i in range(len(self.data_shape)):
            self.data_shape[i] = int(self.data_shape[i])


class InputQueue(API):
    def __init__(self):
        super().__init__()
        self.c, self.h, self.w = None, None, None
        self.stream_name = "serving_stream"

        # TODO: these params can be read from config in future
        self.input_threshold = 0.6
        self.interval_if_error = 1
        self.data_shape_check()

    def data_shape_check(self):
        for num in self.data_shape:
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

    def enqueue(self, uri, **data):
        sink = pa.BufferOutputStream()
        field_list = []
        data_list = []
        for key, value in data.items():
            field, data = get_field_and_data(key, value)
            field_list.append(field)
            data_list.append(data)

        schema = pa.schema(field_list)
        batch = pa.RecordBatch.from_arrays(
            data_list, schema)

        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        buf = sink.getvalue()
        b = buf.to_pybytes()
        b64str = self.base64_encode_image(b)
        d = {"uri": uri, "data": b64str}
        self.__enqueue_data(d)

    def enqueue_tensor(self, uri, data):
        """
        deprecated
        """
        if isinstance(data, np.ndarray):
            # tensor
            data = [data]
        if not isinstance(data, list):
            raise Exception("Your input is invalid, only List of ndarray and ndarray are allowed.")

        sink = pa.BufferOutputStream()
        writer = None
        for d in data:
            shape = np.array(d.shape)
            d = d.astype("float32").flatten()

            data_field = pa.field("data", pa.list_(pa.float32()))
            shape_field = pa.field("shape", pa.list_(pa.int64()))
            tensor_type = pa.struct([data_field, shape_field])

            tensor = pa.array([{'data': d}, {'shape': shape}],
                              type=tensor_type)

            tensor_field = pa.field(uri, tensor_type)
            schema = pa.schema([tensor_field])

            batch = pa.RecordBatch.from_arrays(
                [tensor], schema)
            if writer is None:
                # initialize
                writer = pa.RecordBatchFileWriter(sink, batch.schema)
            writer.write_batch(batch)

        writer.close()
        buf = sink.getvalue()
        b = buf.to_pybytes()
        tensor_encoded = self.base64_encode_image(b)
        d = {"uri": uri, "data": tensor_encoded}
        self.__enqueue_data(d)

    def __enqueue_data(self, data):
        inf = self.db.info()
        try:
            if inf['used_memory'] >= inf['maxmemory'] * self.input_threshold:
                raise redis.exceptions.ConnectionError
            self.db.xadd(self.stream_name, data)
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
    def __init__(self):
        super().__init__()

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
