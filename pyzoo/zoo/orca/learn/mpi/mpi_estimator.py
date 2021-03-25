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

import os
import types
import numpy as np
import subprocess
import cloudpickle
from pyspark.sql import DataFrame
from torch.utils.data import Dataset, DataLoader
from zoo.util.utils import get_node_ip
from zoo.orca.learn.mpi.mpi_runner import MPIRunner


class MPIEstimator:
    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator,
                 scheduler_creator,
                 config=None,
                 init_func=None,  # Init the distributed environment for MPI if any
                 hosts=None,
                 workers_per_node=1,
                 env=None):
        self.dir = os.getcwd()
        self.mpi_runner = MPIRunner(hosts=hosts, processes_per_node=workers_per_node, env=env)
        with open("saved_mpi_estimator.pkl", "wb") as f:
            cloudpickle.dump(
                (model_creator, optimizer_creator, loss_creator, scheduler_creator, config, init_func), f)
        for host in self.mpi_runner.remote_hosts:
            p = subprocess.Popen(["scp", "saved_mpi_estimator.pkl",
                                  "root@{}:{}/".format(host, self.dir)])
            os.waitpid(p.pid, 0)

    # Specify feature_cols and label_cols for Spark DataFrame data.
    # Specify train_func or validate_func for customized training and validation logic.
    # Specify train_batches and validate_batches in case of unbalance data.
    # Specify validate_steps to validate periodically. Note that validation would always be triggered at
    # the end of an epoch.
    def fit(self, data, epochs=1, batch_size=32, validation_data=None, validate_batch_size=32,
            train_func=None, validate_func=None, train_batches=None, validate_batches=None,
            validate_steps=None, feature_cols=None, label_cols=None):
        if isinstance(data, DataFrame):
            assert feature_cols is not None and label_cols is not None, \
                "feature_cols and label_cols must be provided if data is a Spark DataFrame"
            data = data.rdd.map(convert_row(feature_cols, label_cols))
            # TODO: make object store memory configurable?
            object_store_address = self.mpi_runner.launch_plasma(object_store_memory="100g")
            # partition_id, subpartition_id, subpartition_size, object_id, node_ip
            plasma_meta = data.mapPartitionsWithIndex(
                put_to_plasma(object_store_address)).collect()
            train_size_map = {}
            for partition_id, subpartition_id, subpartition_size, object_id, ip in plasma_meta:
                if ip not in train_size_map:
                    train_size_map[ip] = {}
                if partition_id not in train_size_map[ip]:
                    train_size_map[ip][partition_id] = []
                train_size_map[ip][partition_id].append(subpartition_size)
            size = 0
            count = 0
            for node, data in train_size_map.items():
                for partition_id, subpartition_size in data.items():
                    size += sum(subpartition_size)
                    count += len(subpartition_size)
                print("Node {} has {} subpartitions and {} train records".format(node, count, size))
                size = 0
                count = 0
            data_creator = plasma_data_creator(plasma_meta, object_store_address,
                                               self.mpi_runner.processes_per_node, batch_size)
            data.unpersist()
            if validation_data:
                assert isinstance(validation_data, DataFrame)
                validation_data = validation_data.rdd.map(convert_row(feature_cols, label_cols))
                validate_plasma_meta = validation_data.mapPartitionsWithIndex(
                    put_to_plasma(object_store_address)).collect()
                validation_data_creator = plasma_data_creator(
                    validate_plasma_meta, object_store_address,
                    self.mpi_runner.processes_per_node, validate_batch_size)
                validation_data.unpersist()
            else:
                validation_data_creator = None
        else:
            assert isinstance(data, types.FunctionType)
            data_creator = data
            if validation_data:
                assert isinstance(validation_data, types.FunctionType)
                validation_data_creator = validation_data
            else:
                validation_data_creator = None
        if not train_func:
            train_func = train_epoch
        if validation_data_creator:
            if not validate_func:
                validate_func = validate

        with open("mpi_train_data.pkl", "wb") as f:
            cloudpickle.dump((data_creator, epochs, validation_data_creator,
                              train_func, validate_func, train_batches,
                              validate_batches, validate_steps), f)
        for host in self.mpi_runner.remote_hosts:
            p = subprocess.Popen(["scp", "mpi_train_data.pkl",
                                  "root@{}:{}/".format(host, self.dir)])
            os.waitpid(p.pid, 0)
        self.mpi_runner.run(os.path.abspath(__file__ + "/../mpi_train.py"), pkl_path=self.dir)
        self.mpi_runner.shutdown_plasma()

    def shutdown(self):
        self.mpi_runner.shutdown_plasma()


def convert_row(feature_cols, label_cols):
    def convert_for_cols(row, cols):
        result = []
        for name in cols:
            result.append(row[name])
        if len(result) == 1:
            return result[0]
        return result

    def transform(row):
        features = convert_for_cols(row, feature_cols)
        if label_cols:
            labels = convert_for_cols(row, label_cols)
            return features, labels
        else:
            return features,
    return transform


def put_to_plasma(address):

    def process_buffer(buffer):
        import random
        random.shuffle(buffer)  # TODO: Make shuffle configurable?
        buffer_x = [record[0] for record in buffer]
        buffer_y = [record[1] for record in buffer]
        res_buffer = dict()
        if isinstance(buffer_x[0], list):
            res_x = []
            for i in range(len(buffer_x[0])):
                res_x.append(np.array([record[i] for record in buffer_x]))
            res_buffer["x"] = res_x
        else:
            res_buffer["x"] = np.array(buffer_x)
        if isinstance(buffer_y[0], list):
            res_y = []
            for i in range(len(buffer_x[0])):
                res_y.append(np.array([record[i] for record in buffer_y]))
            res_buffer["y"] = res_y
        else:  # TODO: int features and label of type int32?
            res_buffer["y"] = np.array(buffer_y)
        return res_buffer

    def f(index, iterator):
        import pyarrow.plasma as plasma
        client = plasma.connect(address)
        part_size = 1000000  # TODO: Make subpartition size configurable?
        buffer = []
        sub_index = 0
        for record in iterator:
            if len(buffer) == part_size:
                res_buffer = process_buffer(buffer)
                object_id = client.put(res_buffer)  # TODO: check memory usage
                buffer = [record]
                yield index, sub_index, part_size, object_id, get_node_ip()
                sub_index += 1
            else:
                buffer.append(record)
        remain_size = len(buffer)
        if remain_size > 0:
            res_buffer = process_buffer(buffer)
            object_id = client.put(res_buffer)
            buffer = []
            client.disconnect()
            yield index, sub_index, remain_size, object_id, get_node_ip()
        else:
            client.disconnect()

    return f


class PlasmaNDArrayDataset(Dataset):
    def __init__(self, meta_data, object_store_address, workers_per_node=1, batch_size=1):
        import pyarrow.plasma as plasma
        self.client = plasma.connect(object_store_address)
        print("Connected to plasma")

        # All the subpartitions on this node
        all_data = [subpartition for subpartition in meta_data if subpartition[4] == get_node_ip()]
        rank = int(os.environ.get("PMI_RANK", 0))
        # rank = int(os.environ.get("PMIX_RANK", 0))  # For OpenMPI
        local_rank = rank % workers_per_node
        print("Local rank: ", local_rank)
        data_splits = list(chunks(all_data, len(all_data) // workers_per_node))
        worker_data = data_splits[local_rank]
        if len(data_splits) == (workers_per_node + 1):  # Can't evenly split among workers
            remain_data = data_splits[-1]
            if local_rank < len(remain_data):
                worker_data += [remain_data[local_rank]]
        self.object_ids = [subpartition[3] for subpartition in worker_data]
        self.sizes = [subpartition[2] for subpartition in worker_data]
        print("Data size for worker: ", sum(self.sizes))
        self.batch_size = batch_size
        offsets = []
        for i in self.sizes:
            if len(offsets) == 0:
                offsets.append(i)
            else:
                offsets.append(offsets[-1] + i)
        self.offsets = offsets
        self.current_index = 0  # Current index for object_id; data loaded
        self.load_from_plasma(self.current_index)

    def reset(self):
        self.current_index = 0
        self.load_from_plasma(self.current_index)

    def load_from_plasma(self, index):
        print("Loading {}".format(self.object_ids[index]))
        current_data = self.client.get(self.object_ids[index], timeout_ms=0)
        self.current_x = current_data["x"]
        self.current_y = current_data["y"]
        self.current_offset = self.offsets[index]

    def __len__(self):
        return sum(self.sizes) // self.batch_size

    def __getitem__(self, i):  # Directly get a batch
        # print("Loading batch ", i)
        if i == 0 and self.current_index != 0:
            self.reset()
        current_available_size = self.current_offset - i * self.batch_size
        x_list = []
        y_list = []
        if current_available_size < self.batch_size:
            if current_available_size != 0:
                # Add all the remaining records into this batch
                x_list.append(index(self.current_x, start=-current_available_size))
                y_list.append(index(self.current_y, start=-current_available_size))
            # Load subsequent file(s) to complete the batch
            remain_size = self.batch_size - current_available_size
            while True:
                self.current_index += 1
                self.load_from_plasma(self.current_index)
                if self.sizes[self.current_index] >= remain_size:
                    x_list.append(index(self.current_x, end=remain_size))
                    y_list.append(index(self.current_y, end=remain_size))
                    break
                else:
                    x_list.append(self.current_x)
                    y_list.append(self.current_y)
                    remain_size -= self.sizes[self.current_index]
                    if remain_size == 0:
                        break
        # The current file contains a full batch
        elif current_available_size == self.batch_size:
            x_list.append(index(self.current_x, start=-current_available_size))
            y_list.append(index(self.current_y, start=-current_available_size))
        else:
            x_list.append(index(self.current_x, start=-current_available_size, end=-current_available_size + self.batch_size))
            y_list.append(index(self.current_y, start=-current_available_size, end=-current_available_size + self.batch_size))

        if isinstance(self.current_x, list):
            x_np = []
            for i in range(len(self.current_x)):
                x_np.append(np.concatenate([x[i] for x in x_list]))
        else:
            x_np = np.concatenate(x_list)
        y_np = np.concatenate(y_list)
        # Can put collate_fn into train_func if necessary.
        return x_np, y_np
        # assert X_int_np.shape == (self.batch_size, 13)
        # assert X_cat_np.shape == (self.batch_size, 26)
        # assert y_np.shape == (self.batch_size,)
        # X_int = torch.tensor(X_int_np, dtype=torch.float)
        # X_cat = torch.tensor(X_cat_np, dtype=torch.long)
        # T = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
        #
        # batch_size = X_cat.shape[0]
        # feature_count = X_cat.shape[1]
        # lS_i = X_cat.t()
        # lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)
        # return X_int, lS_o, lS_i, T


def plasma_data_creator(meta_data, object_store_address, workers_per_node=1, batch_size=1):

    def create_plasma_dataloader(config):
        dataset = PlasmaNDArrayDataset(meta_data, object_store_address, workers_per_node, batch_size)
        # TODO: support more options
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=None,
        )
        return loader

    return create_plasma_dataloader


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def index(x, start=None, end=None):
    if isinstance(x, list):
        return [index_numpy(x_i, start, end) for x_i in x]
    else:
        return index_numpy(x, start, end)


def index_numpy(x, start=None, end=None):
    if start:
        if end:
            return x[start:end]
        else:
            return x[start:]
    else:
        if end:
            return x[:end]
        else:
            return x


def train_epoch(model, train_ld, train_batches, optimizer, loss, scheduler):
    train_iter = iter(train_ld)
    for j in range(train_batches):
        if j > 0 and j % len(train_ld) == 0:  # For the case where there are not enough batches.
            train_iter = iter(train_ld)
        x, y = next(train_iter)
        o = model(x, y)
        l = loss(o, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()


def validate(model, valid_ld, validate_batches):
    valid_iter = iter(valid_ld)
    for j in range(validate_batches):
        if j > 0 and j % len(validate_batches) == 0:  # For the case where there are not enough batches.
            valid_iter = iter(valid_ld)
        x, y = next(valid_iter)
        o = model(x, y)
        # TODO: Compute accuracy or add metrics
        return
