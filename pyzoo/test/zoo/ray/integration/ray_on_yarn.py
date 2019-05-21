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
from zoo.ray.util.rayrunner import RayRunner

slave_num = 2
rayrunner = RayRunner.from_spark(hadoop_conf="/home/zhichao/god/yarn35_config",
                     slave_num=slave_num,
                     slave_cores=28,
                     slave_memory="60g",
                     driver_cores=4,
                     driver_memory="10g",
                     conda_name="ray36",
                     extra_pmodule_zip="/home/zhichao/god/rayonspark/pyzoo/zoo.zip",
                     force_purge=True,
                     verbose=True)
rayrunner.start()

@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()

actors = [TestRay.remote() for i in range(0, slave_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
print([ray.get(actor.ip.remote()) for actor in actors])

rayrunner.stop()


