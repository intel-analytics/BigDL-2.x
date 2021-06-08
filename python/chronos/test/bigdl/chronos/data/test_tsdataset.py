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

import pytest
import numpy as np
import pandas as pd
import random

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.chronos.data import TSDataset

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def get_multi_id_ts_df():
    sample_num = 100
    train_df = pd.DataFrame({"value": np.random.randn(sample_num),
                             "id": np.array(['00']*50 + ['01']*50),
                             "extra feature": np.random.randn(sample_num)})
    train_df["datetime"] = pd.date_range('1/1/2019', periods=sample_num)
    train_df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    return train_df


def get_ugly_ts_df():
    data = np.random.random_sample((100, 5))
    mask = np.random.random_sample((100, 5))
    mask[mask >= 0.4] = 2
    mask[mask < 0.4] = 1
    mask[mask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan  # make sure column 'a' has a N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=100)
    df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    df["id"] = np.array(['00']*50 + ['01']*50)
    return df


class TestTSDataset(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_tsdataset_initialization(self):
        df = get_ts_df()

        # legal input
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        assert tsdata._id_list == ['00']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                       extra_feature_col="extra feature", id_col="id")
        assert tsdata._id_list == ['00']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df.drop(columns=["id"]), dt_col="datetime",
                                       target_col=["value"], extra_feature_col="extra feature")
        assert tsdata._id_list == ['0']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        # illegal input
        with pytest.raises(AssertionError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature", id_col=0)
        with pytest.raises(AssertionError):
            tsdata = TSDataset.from_pandas(df, dt_col=0, target_col=["value"],
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(AssertionError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=0,
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(AssertionError):
            tsdata = TSDataset.from_pandas(0, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(AssertionError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value1"],
                                           extra_feature_col="extra feature", id_col="id")

    def test_tsdataset_roll_single_id(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        with pytest.raises(RuntimeError):
            tsdata.to_numpy()

        # roll train
        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=["extra feature"], target_col="value")
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=[], target_col="value")
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 1)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        # roll test
        horizon = 0
        lookback = random.randint(1, 20)

        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y is None
        tsdata._check_basic_invariants()

    def test_tsdataset_roll_multi_id(self):
        df = get_multi_id_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # roll train
        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1), lookback, 4)
        assert y.shape == ((50-lookback-horizon+1), horizon, 2)
        tsdata._check_basic_invariants()

    def test_tsdataset_imputation(self):
        df = get_ugly_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="e",
                                       extra_feature_col=["a", "b", "c", "d"], id_col="id")
        tsdata.impute(mode="last")
        assert tsdata.to_pandas().isna().sum().sum() == 0
        assert len(tsdata.to_pandas()) == 100
        tsdata._check_basic_invariants()

    def test_tsdataset_deduplicate(self):
        df = get_ugly_ts_df()
        for i in range(20):
            df.loc[len(df)] = df.loc[np.random.randint(0, 99)]
        assert len(df) == 120
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="e",
                                       extra_feature_col=["a", "b", "c", "d"], id_col="id")
        tsdata.deduplicate()
        assert len(tsdata.to_pandas()) == 100
        tsdata._check_basic_invariants()

    def test_tsdataset_datetime_feature(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_dt_feature()
        assert set(tsdata.to_pandas().columns) == {'IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   'DAY(datetime)',
                                                   'IS_WEEKEND(datetime)',
                                                   'WEEKDAY(datetime)',
                                                   'MONTH(datetime)',
                                                   'DAYOFYEAR(datetime)',
                                                   'WEEKOFYEAR(datetime)',
                                                   'MINUTE(datetime)',
                                                   'extra feature',
                                                   'value',
                                                   'datetime',
                                                   'id'}
        assert set(tsdata.feature_col) == {'IS_AWAKE(datetime)',
                                           'IS_BUSY_HOURS(datetime)',
                                           'HOUR(datetime)',
                                           'DAY(datetime)',
                                           'IS_WEEKEND(datetime)',
                                           'WEEKDAY(datetime)',
                                           'MONTH(datetime)',
                                           'DAYOFYEAR(datetime)',
                                           'WEEKOFYEAR(datetime)',
                                           'MINUTE(datetime)',
                                           'extra feature'}
        tsdata._check_basic_invariants()

    def test_tsdataset_scale_unscale(self):
        df = get_ts_df()
        df_test = get_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata_test = TSDataset.from_pandas(df_test, dt_col="datetime", target_col="value",
                                            extra_feature_col=["extra feature"], id_col="id")

        from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        scalers = [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]
        for scaler in scalers:
            tsdata.scale(scaler)
            tsdata_test.scale(scaler, fit=False)

            with pytest.raises(AssertionError):
                assert_frame_equal(tsdata.to_pandas(), df)
            with pytest.raises(AssertionError):
                assert_frame_equal(tsdata_test.to_pandas(), df_test)

            tsdata.unscale()
            tsdata_test.unscale()

            assert_frame_equal(tsdata.to_pandas(), df)
            assert_frame_equal(tsdata_test.to_pandas(), df_test)

        tsdata._check_basic_invariants()

    def test_tsdataset_unscale_numpy(self):
        df = get_multi_id_ts_df()
        df_test = get_multi_id_ts_df()

        from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        scalers = [StandardScaler(),
                   StandardScaler(with_mean=False),
                   StandardScaler(with_std=False),
                   MaxAbsScaler(),
                   MinMaxScaler(),
                   MinMaxScaler(feature_range=(1, 3)),
                   RobustScaler(),
                   RobustScaler(with_centering=False),
                   RobustScaler(with_scaling=False),
                   RobustScaler(quantile_range=(20, 80))]

        for scaler in scalers:
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                           extra_feature_col=["extra feature"], id_col="id")
            tsdata_test = TSDataset.from_pandas(df_test, dt_col="datetime", target_col="value",
                                                extra_feature_col=["extra feature"], id_col="id")
            tsdata.gen_global_feature(settings="minimal")\
                  .gen_dt_feature()\
                  .scale(scaler)\
                  .roll(lookback=5, horizon=4, id_sensitive=True)
            tsdata_test.gen_global_feature(settings="minimal")\
                       .gen_dt_feature()\
                       .scale(scaler, fit=False)\
                       .roll(lookback=5, horizon=4, id_sensitive=True)

            _, _ = tsdata.to_numpy()
            _, y_test = tsdata_test.to_numpy()

            pred = np.copy(y_test)  # sanity check

            unscaled_pred = tsdata._unscale_numpy(pred)
            unscaled_y_test = tsdata._unscale_numpy(y_test)
            tsdata_test.unscale()\
                       .roll(lookback=5, horizon=4, id_sensitive=True)
            _, unscaled_y_test_reproduce = tsdata_test.to_numpy()

            assert_array_almost_equal(unscaled_pred, unscaled_y_test_reproduce)
            assert_array_almost_equal(unscaled_y_test, unscaled_y_test_reproduce)

            tsdata._check_basic_invariants()

    def test_tsdataset_resample(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.resample('2D', df["datetime"][0], df["datetime"][99])
        assert len(tsdata.to_pandas()) == 50
        tsdata._check_basic_invariants()

    def test_tsdataset_split(self):
        df = get_multi_id_ts_df()
        tsdata_train, tsdata_valid, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                  extra_feature_col=["extra feature"], id_col="id",
                                  with_split=True, val_ratio=0.1, test_ratio=0.1,
                                  largest_look_back=5, largest_horizon=2)

        assert set(np.unique(tsdata_train.to_pandas()["id"])) == {"00", "01"}
        assert set(np.unique(tsdata_valid.to_pandas()["id"])) == {"00", "01"}
        assert set(np.unique(tsdata_test.to_pandas()["id"])) == {"00", "01"}

        assert len(tsdata_train.to_pandas()) == (50 * 0.8)*2
        assert len(tsdata_valid.to_pandas()) == (50 * 0.1 + 5 + 2 - 1)*2
        assert len(tsdata_test.to_pandas()) == (50 * 0.1 + 5 + 2 - 1)*2

    def test_tsdataset_global_feature(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_global_feature(settings="minimal")
        tsdata._check_basic_invariants()
