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
import os
import pytest
import pandas as pd
import random

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.chronos.data.repo_dataset import get_public_dataset


class TestRepoDataset(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_init_dataset(self):
        name = random.sample([x for x in range(10)], 5)
        path = '~/.chronos/dataset'
        with pytest.raises(AssertionError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'nyc_taxi'
        path = random.sample([x for x in range(10)], 5)
        with pytest.raises(AssertionError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'chronos_dataset'
        path = '~/.chorons/dataset/'
        with pytest.raises(NameError):
            get_public_dataset(name, path=path, redownload=False)

    def test_network_traffic(self):
        name = 'network_traffic'
        path = '~/.chronos/dataset/'
        download_list = [f'2018{val:02d}.agr' for val in range(1, 13)] + \
                        [f'2019{val:02d}.agr' for val in range(1, 13)]
        lookback, horizon = 6, 1
        tsdata = get_public_dataset(name, path=path, redownload=False)

        # issubset
        exists_file = os.listdir(os.path.expanduser(os.path.join(path, name)))
        assert set(download_list).issubset(exists_file)

        raw_df = pd.read_csv(os.path.join(os.path.join(path, name), name+'_data.csv'))
        assert raw_df.shape == (8760, 4)
        assert list(raw_df.columns) == ['StartTime', 'EndTime', 'AvgRate', 'total']

        x, y = tsdata.roll(lookback=lookback, horizon=horizon).to_numpy()
        assert x.shape == (raw_df.shape[0]-lookback-horizon+1, lookback, 2)
        assert y.shape == (raw_df.shape[0]-lookback-horizon+1, horizon, 2)

        # with_split, redownload=True
        tsdata_train, _,\
            tsdata_test = get_public_dataset(name, path=path,
                                             redownload=False,
                                             with_split=True,
                                             test_ratio=0.1)
        for tsdata in [tsdata_train, tsdata_test]:
            tsdata.roll(lookback=lookback, horizon=horizon)

        x, y = tsdata_train.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.9)-lookback-horizon+1, lookback, 2)
        assert y.shape == (int(raw_df.shape[0]*0.9)-lookback-horizon+1, horizon, 2)

        x, y = tsdata_test.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, lookback, 2)
        assert y.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, horizon, 2)

    def test_AIOps(self):
        name = 'AIOps'
        path = '~/.chronos/dataset/'
        download_list = ['AIOps_data.csv', 'machine_usage.csv', 'machine_usage.tar.gz']
        lookback, horizon = 60, 1

        tsdata = get_public_dataset(name, path, redownload=False)
        # issubset
        exists_file = os.listdir(os.path.expanduser(os.path.join(path, name)))
        assert set(download_list).issubset(exists_file)

        raw_df = pd.read_csv(os.path.join(os.path.join(path, name), name+'_data.csv'))
        assert raw_df.shape == (61570, 4)
        assert list(raw_df.columns) == ['id', 'time_step', 'cpu_usage', 'mem_usage']

        # tsdata.
        x, y = tsdata.roll(lookback=lookback, horizon=horizon).to_numpy()
        assert x.shape == (tsdata.df.shape[0]-lookback-horizon+1, lookback, 1)
        assert y.shape == (tsdata.df.shape[0]-lookback-horizon+1, horizon, 1)

        # with_split
        tsdata_train,\
            tsdata_val,\
            tsdata_test = get_public_dataset(name, path=path,
                                             redownload=False,
                                             with_split=True,
                                             val_ratio=0.1,
                                             test_ratio=0.1)
        from sklearn.preprocessing import StandardScaler
        stand = StandardScaler()

        for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
            tsdata.scale(stand, fit=(tsdata is tsdata_train))\
                  .roll(lookback=lookback, horizon=horizon)

        x, y = tsdata_train.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+1, horizon, 1)

        x, y = tsdata_val.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, horizon, 1)

        x, y = tsdata_test.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, horizon, 1)

    def test_fsi(self):
        name = 'fsi'.lower()
        path = '~/.chronos/dataset/'
        download_list = ['fsi_data.csv', 'individual_stocks_5yr', 'individual_stocks_5yr.zip']
        lookback, horizon = 50, 1
        tsdata = get_public_dataset(name, path=path, download=False)

        # issubset.
        exists_file = os.listdir(os.path.expanduser(os.path.join(path, name)))
        assert set(download_list).issubset(exists_file)

        # tsdata.
        x, y = tsdata.roll(lookback=lookback, horizon=horizon).to_numpy()
        assert x.shape == (tsdata.df.shape[0]-lookback-horizon+1, lookback, 1)
        assert y.shape == (tsdata.df.shape[0]-lookback-horizon+1, horizon, 1)

        raw_df = pd.read_csv(os.path.join(os.path.join(path, name),
                             name+'_data.csv'),
                             usecols=[0, 5])
        assert raw_df.shape == (1259, 2)
        assert list(raw_df.columns) == ['ds', 'y']

        # redownload=True, with_split.
        tsdata_train, _,\
            tsdata_test = get_public_dataset(name, path=path,
                                             redownload=False,
                                             with_split=True,
                                             test_ratio=0.2)
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler()
        for tsdata in [tsdata_train, tsdata_test]:
            tsdata.scale(minmax, fit=tsdata is tsdata_train)\
                  .roll(lookback=lookback, horizon=horizon)

        x, y = tsdata_train.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+2, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+2, horizon, 1)

        x, y = tsdata_test.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.2)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.2)-lookback-horizon+1, horizon, 1)

    def test_nyc_taxi(self):
        name = 'nyc_taxi'
        path = '~/.chronos/dataset/'
        download_list = ['nyc_taxi.csv', 'nyc_taxi_data.csv']
        lookback, horizon = 6, 1
        tsdata = get_public_dataset(name, path=path, redownload=False)

        # raw_df shape, columns.
        raw_df = pd.read_csv(os.path.join(os.path.join(path, name),
                             download_list[0]),
                             parse_dates=['timestamp'])
        assert raw_df.shape == (10320, 2)
        assert list(raw_df.columns) == ['timestamp', 'value']

        # get_tsdata
        assert tsdata.df.shape == (10320, 3)
        assert list(tsdata.df.columns) == ['timestamp', 'value', 'id']

        # issubset.
        exists_file = os.listdir(os.path.expanduser(os.path.join(path, name)))
        assert set(download_list).issubset(exists_file)

        # roll tsdata.
        x, y = tsdata.roll(lookback=lookback, horizon=horizon).to_numpy()
        assert x.shape == (tsdata.df.shape[0]-lookback-horizon+1, lookback, 1)
        assert y.shape == (tsdata.df.shape[0]-lookback-horizon+1, horizon, 1)

        # redownload=True, with_split
        tsdata_train,\
            tsdata_val,\
            tsdata_test = get_public_dataset(name, path=path,
                                             redownload=True,
                                             with_split=True,
                                             val_ratio=0.1,
                                             test_ratio=0.1)

        exists_file = os.listdir(os.path.expanduser(os.path.join(path, name)))
        assert set(download_list).issubset(exists_file)

        from sklearn.preprocessing import StandardScaler
        stand = StandardScaler()
        for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
            tsdata.scale(stand, fit=tsdata is tsdata_train)\
                  .roll(lookback=lookback, horizon=horizon)

        x, y = tsdata_train.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.8)-lookback-horizon+1, horizon, 1)
        x, y = tsdata_val.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, horizon, 1)
        x, y = tsdata_test.to_numpy()
        assert x.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, lookback, 1)
        assert y.shape == (int(raw_df.shape[0]*0.1)-lookback-horizon+1, horizon, 1)
