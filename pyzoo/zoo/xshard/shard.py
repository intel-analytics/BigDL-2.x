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
    def __init__(self, shards):
        pass

    def apply(self, func):
        pass

    def collect(self):
        pass
    

class RayDataShards(DataShards):
    def __init__(self, shard_list):
        self.shard_list = shard_list
        self.object_ids = None
        self.partition_id_list = None

    def apply(self, func, *args):
        [shard.apply.remote(func, *args) for shard in self.shard_list]
        return self

    def collect(self):
        return ray.get([shard.get_data.remote() for shard in self.shard_list])

    def collect_ids(self):
        result_ids = [shard.get_data.remote() for shard in self.shard_list]
        # done_ids, undone_ids = ray.wait(result_ids, num_returns=len(self.shard_list))
        # assert len(undone_ids) == 0
        self.object_ids = result_ids
        return result_ids

    def repartition(self, num_partitions):
        if self.object_ids:
            self.partition_id_list = list(chunk(self.object_ids, num_partitions))
        else:
            self.partition_id_list = list(chunk(self.collect_ids(), num_partitions))
        return self

    def get_partitions(self):
        return self.partition_id_list
