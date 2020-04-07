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

import zoo.xshard.pandas
from zoo.common.nncontext import init_nncontext


def rename(df):
    df = df.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                            'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                            'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                            'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})
    return df

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", type=str, dest="file_path",
                      help="The file path to be read")

    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext()

    # read data
    file_path = options.file_path
    data_shard = zoo.xshard.pandas.read_csv(file_path, sc)
    data = data_shard.collect()

    # repartition
    data_shard.repartition(2)

    # apply function on each element
    data_shards_2 = data_shard.apply(rename)
    data2 = data_shard.collect()

    sc.stop()

