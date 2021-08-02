
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

import sys
from optparse import OptionParser

import zoo.orca.data.pandas
from zoo.orca import init_orca_context, stop_orca_context


def process_feature(df, awake_begin=6, awake_end=23):
    import pandas as pd
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['datetime'].dt.hour
    df['awake'] = (((df['hours'] >= awake_begin) & (df['hours'] <= awake_end))
                   | (df['hours'] == 0)).astype(int)
    return df


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", type=str, dest="file_path",
                      help="The file path to be read")

    (options, args) = parser.parse_args(sys.argv)
#    sc = init_orca_context(cores="*", memory="4g")
    sc = init_orca_context(cluster_mode="k8s", master="k8s://https://127.0.0.1:8443", container_image="10.239.45.10/arda/intelanalytics/hyper-zoo:0.11.0", num_nodes=2, memory="50g", extra_executor_memory_for_ray="100g", cores=4, conf={"spark.driver.host": "172.16.0.200", "spark.driver.port": "54321","spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "persistent-volume-claim", "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/zoo"})


    # read data
    file_path = options.file_path
    data_shard = zoo.orca.data.pandas.read_csv(file_path)
    data = data_shard.collect()

    # repartition
    data_shard = data_shard.repartition(2)

    # apply function on each element
    trans_data_shard = data_shard.transform_shard(process_feature)
    data2 = trans_data_shard.collect()

    stop_orca_context()
