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

from optparse import OptionParser
import sys
from zoo import init_spark_on_local, init_spark_on_yarn
import zoo.xshard.pandas
from zoo.ray.util.raycontext import RayContext


def negative(df, column_name):
    df[column_name] = df[column_name] * (-1)
    return df


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", type=str, dest="file_path",
                      help="The file path to be read")

    (options, args) = parser.parse_args(sys.argv)

    sc = init_spark_on_local(cores="*")

    ray_ctx = RayContext(sc=sc,
                        object_store_memory="10g",
                        env={"AWS_ACCESS_KEY_ID":"AKIA2TEO6PMT6U3JVYFC",
                          "AWS_SECRET_ACCESS_KEY":"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"}
                         )

    ray_ctx.init(object_store_memory="10g")

    # read data
    file_path = options.file_path
    data_shard = zoo.xshard.pandas.read_json(file_path, ray_ctx)

    # collect object ids for data
    ids = data_shard.collect_ids()
    print("get ids:")
    print(ids)

    # repartition
    data_shard.repartition(2)
    repartitioned_ids = data_shard.get_partitions()
    print("get repartitioned ids:")
    print(repartitioned_ids)

    # collect data
    data = data_shard.collect()
    print("collected data : %s" % data[0].iloc[0])

    # apply function on each element
    data_shards_2 = data_shard.apply(negative, "label")
    ids2 = data_shards_2.collect_ids()
    print("get ids 2 :")
    print(ids2)
    data2 = data_shards_2.collect()
    print("collected data : %s" % data2[0].iloc[0])

