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

class DataShard(object):
    def __init__(self, shards):
        pass

    def apply(self, func):
        pass


    def collect_data(self):
        pass


    def get_shards(self):
        pass


class RayDataShard(DataShard):
    def __init__(self, shard_list):
        self.shard_list = shard_list

    def apply(self, func):
        return [shard.apply.remote(func) for shard in self.shard_list]

    def collect_data(self):
        return ray.get([shard.get_data.remote() for shard in self.shard_list])

    def get_shards(self):
        return [shard.get_data.remote() for shard in self.shard_list]
