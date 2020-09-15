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

import redis
import time
from zoo.serving.schema import *
import httpx
import json

RESULT_PREFIX = "cluster-serving_"


def perdict(frontend_url, request_str):
    httpx.post(frontend_url + "/predict", data=request_str)


class API:
    """
    base level of API control
    select data pipeline here, Redis/Kafka/...
    interface preserved for API class
    """
    def __init__(self, host=None, port=None, name="serving_stream"):
        self.name = name
        if not host:
            host = "localhost"
        if not port:
            port = "6379"

        self.db = redis.StrictRedis(host=host,
                                    port=port, db=0)
        try:
            self.db.xgroup_create(name, "serving")
        except Exception:
            print("redis group exist, will not create new one")


class InputQueue(API):
    def __init__(self, host=None, port=None, sync=False, frontend_url=None):
        super().__init__(host, port)
        self.sync = sync
        self.frontend_url = frontend_url
        if self.sync:
            try:
                res = httpx.get(frontend_url)
                if res.status_code == 200:
                    httpx.PoolLimits(max_keepalive=1, max_connections=1)
                    self.cli = httpx.Client()
                    print("Attempt connecting to Cluster Serving frontend success")
                else:
                    raise ConnectionError()
            except Exception as e:
                print("Connection error, please check your HTTP server. Error msg is ", e)

        # TODO: these params can be read from config in future
        self.input_threshold = 0.6
        self.interval_if_error = 1

    def predict(self, request_str):
        """
        Sync API, block waiting until get response
        :return:
        """
        response = self.cli.post(self.frontend_url + "/predict", data=request_str)
        predictions = json.loads(response.text)['predictions']
        processed = predictions[0].lstrip("{value=").rstrip("}")
        return processed

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
            self.db.xadd(self.name, data)
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
    def __init__(self, host=None, port=None):
        super().__init__(host, port)

    def dequeue(self):
        res_list = self.db.keys(RESULT_PREFIX + self.name + ':*')
        decoded = {}
        for res in res_list:
            res_dict = self.db.hgetall(res.decode('utf-8'))
            res_id = res.decode('utf-8').split(":")[1]
            res_value = res_dict[b'value'].decode('utf-8')
            decoded[res_id] = res_value
            self.db.delete(res)
        return decoded

    def query_and_delete(self, uri):
        self.query(uri, True)

    def query(self, uri, delete=False):
        res_dict = self.db.hgetall(RESULT_PREFIX + self.name + ':' + uri)

        if not res_dict or len(res_dict) == 0:
            return "[]"
        if delete:
            self.db.delete(RESULT_PREFIX + self.name + ':' + uri)
        s = res_dict[b'value'].decode('utf-8')
        if s == "NaN":
            return s
        return self.get_ndarray_from_b64(s)

    def get_ndarray_from_b64(self, b64str):
        b = base64.b64decode(b64str)
        a = pa.BufferReader(b)
        c = a.read_buffer()
        myreader = pa.ipc.open_stream(c)
        r = [i for i in myreader]
        assert len(r) > 0
        if len(r) == 1:
            return self.get_ndarray_from_record_batch(r[0])
        else:
            l = []
            for ele in r:
                l.append(self.get_ndarray_from_record_batch(ele))
            return l

    def get_ndarray_from_record_batch(self, record_batch):
        data = record_batch[0].to_numpy()
        shape_list = record_batch[1].to_pylist()
        shape = [i for i in shape_list if i]
        return data.reshape(shape)
