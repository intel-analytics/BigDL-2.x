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

import ray

from zoo.xshard.utils import *


class DataShards(object):
    """
    A collection of data which can be pre-processed parallelly.
    """
    def apply(self, func, *args):
        """
        Appy function on each element in the DataShards
        :param func: pre-processing function
        :param args: arguments for the pre-processing function
        :return: this DataShard
        """
        pass

    def collect(self):
        """
        Returns a list that contains all of the elements in this DataShards
        :return: list of elements
        """
        pass


class RayDataShards(DataShards):
    """
    A collection of data which can be pre-processed parallelly on Ray
    """

    def __init__(self, partitions):
        self.partitions = partitions
        self.shard_list = flatten([partition.shard_list for partition in partitions])

    def apply(self, func, *args):
        """
        Appy function on each element in the DataShards
        :param func: pre-processing function.
        In the function, the element object should be the first argument
        :param args: rest arguments for the pre-processing function
        :return: this DataShard
        """
        [shard.apply.remote(func, *args) for shard in self.shard_list]
        return self

    def apply_partition(self, func, *args):
        """
        Appy function on each partition in the DataShards
        :param func: pre-processing function which process data on each partition.
         In the function, partition data object should be the first argument.
        :param args: rest arguments for the pre-processing function
        :return: this DataShard
        """
        for partition in self.partitions:
            new_data = __apply_partition__.remote(partition.get_data(), func, *args)
            partition.data = new_data
        return self

    def collect(self):
        """
        Returns a list that contains all of the elements in this DataShards
        :return: list of elements
        """
        return ray.get([shard.get_data.remote() for shard in self.shard_list])

    def repartition(self, num_partitions):
        """
        Repartition DataShards.
        :param num_partitions: number of partitions
        :return: this DataShards
        """
        shards_partitions = list(chunk(self.shard_list, num_partitions))
        self.partitions = [RayPartition(shards) for shards in shards_partitions]
        return self

    def get_partitions(self):
        """
        Return partition list of the DataShards
        :return: partition list
        """
        return self.partitions


class RayPartition(object):
    """
    Partition of RayDataShards
    """
    def __init__(self, shard_list):
        self.shard_list = shard_list
        done_ids, undone_ids = ray.wait([shard.get_data.remote()
                                         for shard in self.shard_list],
                                        num_returns=len(self.shard_list))
        assert len(undone_ids) == 0
        self.data = done_ids

    def get_data(self):
        """
        Get the content of this partition
        :return:
        """
        return self.data


@ray.remote
def __apply_partition__(data, func, *args):
    elem_iterable = ray.get(data)
    data = func(elem_iterable, *args)
    return data
