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
from zoo.chronos.data.utils.public_dataset import PublicDataset


def get_public_dataset(name="network_traffic",
                       path='~/.chronos/dataset',
                       redownload=False,
                       **kwargs):
    """
    Get public dataset.

    >>> from zoo.chronos.data.repo_dataset import get_public_dataset
    >>> tsdata_network_traffic = get_public_dataset(name="network_traffic")

    :param name: str, public dataset name, e.g. "network traffic".
    :param path: str, download path, the value defatults to "~/.chronos/dataset/".
    :param redownload: bool, if redownload the raw dataset file(s).
    :param **kwargs: dict, extra arguments passed to initialize the tsdataset.
    """
    assert isinstance(name, str) or isinstance(path, str),\
        "Name and path must be string."

    public_dataset = PublicDataset(name=name,
                                   path=path,
                                   redownload=redownload,
                                   **kwargs).get_public_data()
    if name == 'network_traffic':
        return public_dataset.preprocess_network_traffic()\
                             .get_tsdata(dt_col='StartTime',
                                         target_col=['AvgRate', 'total'])
    elif name == 'AIOps':
        return public_dataset.preprocess_AIOps()\
                             .get_tsdata(dt_col='time_step',
                                         target_col=['cpu_usage'])
    elif name == 'fsi':
        return public_dataset.preprocess_fsi()\
                             .get_tsdata(dt_col='ds',
                                         target_col=['y'])
    elif name == 'nyc_taxi':
        return public_dataset.preprocess_nyc_taxi()\
                             .get_tsdata(dt_col='timestamp',
                                         target_col=['value'])
    else:
        raise NameError(f'Only "network_traffic", "AIOps", "fsi", "nyc_taxi"'
                        'are supported in Chronos built-in dataset, while get {name}.')
