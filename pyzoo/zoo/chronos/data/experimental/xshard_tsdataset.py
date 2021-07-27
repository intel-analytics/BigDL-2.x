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


from zoo.orca.data.shard import SparkXShards
from zoo.chronos.data.utils.roll import roll_timeseries_dataframe
from zoo.chronos.data.utils.split import split_timeseries_dataframe
from zoo.chronos.data.experimental.utils import add_row, transform_to_dict

_DEFAULT_ID_COL_NAME = "id"
_DEFAULT_ID_PLACEHOLDER = "0"

class XShardTSDataset:

    def __init__(self, shard, **schema):
        '''
        XShardTSDataset is an abstract of time series dataset.
        Cascade call is supported for most of the transform methods.
        XShardTSDataset will partition the dataset by id_col, which is experimental.
        '''
        self.shard = shard
        self.id_col = schema["id_col"]
        self.dt_col = schema["dt_col"]
        self.feature_col = schema["feature_col"].copy()
        self.target_col = schema["target_col"].copy()

        self.numpy_shard = None

        self._id_list = list(shard[self.id_col].unique())

    @staticmethod
    def from_xshard(shard,
                    dt_col,
                    target_col,
                    id_col=None,
                    extra_feature_col=None,
                    with_split=False,
                    val_ratio=0,
                    test_ratio=0.1):
        '''
        Initialize xshardtsdataset(s) from xshard pandas dataframe.

        :param df: an xshard pandas dataframe for your raw time series data.
        :param dt_col: a str indicates the col name of datetime
               column in the input data frame.
        :param target_col: a str or list indicates the col name of target column
               in the input data frame.
        :param id_col: (optional) a str indicates the col name of dataframe id. If
               it is not explicitly stated, then the data is interpreted as only
               containing a single id.
        :param extra_feature_col: (optional) a str or list indicates the col name
               of extra feature columns that needs to predict the target column.
        :param with_split: (optional) bool, states if we need to split the dataframe
               to train, validation and test set. The value defaults to False.
        :param val_ratio: (optional) float, validation ratio. Only effective when
               with_split is set to True. The value defaults to 0.
        :param test_ratio: (optional) float, test ratio. Only effective when with_split
               is set to True. The value defaults to 0.1.

        :return: a XShardTSDataset instance when with_split is set to False,
                 three XShardTSDataset instances when with_split is set to True.

        Create a xshardtsdataset instance by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> from zoo.orca.data.pandas import read_csv
        >>> shard = read_csv(csv_path)
        >>> tsdataset = XShardTSDataset.from_xshard(shard, dt_col="datetime",
        >>>                                         target_col="value", id_col="id",
        >>>                                         extra_feature_col=["extra feature 1",
        >>>                                                            "extra feature 2"])
        '''

        _check_type(shard, "shard", SparkXShards)

        target_col = _to_list(target_col, name="target_col")
        feature_col = _to_list(extra_feature_col, name="extra_feature_col")

        if id_col is None:
            shard = shard.transform_shard(add_row,
                                          _DEFAULT_ID_COL_NAME,
                                          _DEFAULT_ID_PLACEHOLDER)
            id_col = _DEFAULT_ID_COL_NAME

        # repartition to id
        shard = shard.partition_by(cols=id_col,
                                   num_partitions=len(shard[id_col].unique()))

        if with_split:
            tsdataset_shards\
                = shard.transform_shard(split_timeseries_dataframe,
                                        id_col, val_ratio, test_ratio).split()
            return [XShardTSDataset(shard=tsdataset_shards[i],
                                    id_col=id_col,
                                    dt_col=dt_col,
                                    target_col=target_col,
                                    feature_col=feature_col) for i in range(3)]

        return XShardTSDataset(shard=shard,
                               id_col=id_col,
                               dt_col=dt_col,
                               target_col=target_col,
                               feature_col=feature_col)

    def roll(self,
             lookback,
             horizon,
             feature_col=None,
             target_col=None):
        '''
        Sampling by rolling for machine learning/deep learning models.

        :param lookback: int, lookback value.
        :param horizon: int or list,
               if `horizon` is an int, we will sample `horizon` step
               continuously after the forecasting point.
               if `horizon` is a list, we will sample discretely according
               to the input list.
               specially, when `horizon` is set to 0, ground truth will be generated as None.
        :param feature_col: str or list, indicates the feature col name. Default to None,
               where we will take all available feature in rolling.
        :param target_col: str or list, indicates the target col name. Default to None,
               where we will take all target in rolling. it should be a subset of target_col
               you used to initialize the xshardtsdataset.

        :return: the xshardtsdataset instance.
        '''
        feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
            else self.feature_col
        target_col = _to_list(target_col, "target_col") if target_col is not None \
            else self.target_col
        self.numpy_shard = self.shard.transform_shard(roll_timeseries_dataframe,
                                                None, lookback, horizon, feature_col, target_col)
        return self

    def to_xshard(self):
        '''
        Export rolling result in form of a dict of numpy ndarray {'x': ..., 'y': ...}

        :return: a 2-element dict xshard. each value is a 3d numpy ndarray. The ndarray
                 is casted to float64.
        '''
        return self.numpy_shard.transform_shard(transform_to_dict)


def _to_list(item, name, expect_type=str):
    if isinstance(item, list):
        return item
    if item is None:
        return []
    _check_type(item, name, expect_type)
    return [item]


def _check_type(item, name, expect_type):
    assert isinstance(item, expect_type),\
        f"a {str(expect_type)} is expected for {name} but found {type(item)}"
