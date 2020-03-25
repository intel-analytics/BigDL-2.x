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

from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
from zoo.xshard.pandas import read_csv, read_json

if __name__ == "__main__":

    sc = init_spark_on_local(cores="*")
    #sc = init_spark_on_yarn(
    #    hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
    #    conda_name="mxnet",
    #    # 1 executor for ray head node. The remaining executors for raylets.
    #    # Each executor is given enough cores to be placed on one node.
    #    # Each MXNetRunner will run in one executor, namely one node.
    #    num_executor=2*config["num_workers"],
    #    executor_cores=44,
    #    executor_memory="10g",
    #    driver_memory="2g",
    #    driver_cores=16,
    #    extra_executor_memory_for_ray="5g",
    #    extra_python_lib="mxnet_runner.py")
    ray_ctx = RayContext(sc=sc,
                     object_store_memory="5g",
                     env={"OMP_NUM_THREADS": "1",
                          "KMP_AFFINITY": "granularity=fine,compact,1,0"})
    ray_ctx.init(object_store_memory="5g")

    # read data
    file_path ="/data/rbi/small/tlogFmt1214"
    data_shard = read_json(file_path, ray_ctx)
    shards = data_shard.get_shards()
    print("get shards:")
    print(shards)
    data = data_shard.collect_data()
    print("collected data : %d" % data[0][1])
