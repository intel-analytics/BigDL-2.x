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



# update python in pip: cp -rf ~/god/rayonspark/pyzoo/zoo/* /home/zhichao/anaconda2/envs/ray36/lib/python3.6/site-pac
# This is required if run on pycharm
import sys
sys.path.insert(0, "/root/anaconda2/envs/ray36/lib/python3.6/site-packages/zoo/share/conf/spark-analytics-zoo.conf")

slave_num = 3
rayrunner = RayRunner.from_spark(hadoop_conf="/opt/work/almaren-yarn-config/",
                                 slave_num=slave_num,
                                 slave_cores=28,
                                 slave_memory="60g",
                                 driver_cores=4,
                                 driver_memory="10g",
                                 conda_name="ray36-dev",
                                 extra_pmodule_zip="/opt/work/rayonspark/pyzoo/zoo.zip",
                                 jars="/opt/work/rayonspark/dist/lib/analytics-zoo-bigdl_0.8.0-spark_2.4.0-0.5.0-SNAPSHOT-jar-with-dependencies.jar",
                                 force_purge=False,
                                 verbose=True,
                                 env={"http_proxy": "http://child-prc.intel.com:913",
                                      "http_proxys": "http://child-prc.intel.com:913"})
rayrunner.start()

@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

    # def check_cv2(self):
    # conda install -c conda-forge opencv==3.4.2
    #     import cv2
    #     return cv2.__version__

    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()

    def network(self):
        from urllib.request import urlopen
        try:
            urlopen('http://www.baidu.com', timeout=1)
            return True
        except Exception as err:
            return False


actors = [TestRay.remote() for i in range(0, slave_num)]
print([ray.get(actor.hostname.remote()) for actor in actors])
print([ray.get(actor.ip.remote()) for actor in actors])
print([ray.get(actor.network.remote()) for actor in actors])


rayrunner.stop()


