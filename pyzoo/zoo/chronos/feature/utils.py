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
import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    convert numpy array to list for JSON serialize
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_config(file_path, config, replace=False):
    """
    :param file_path: the file path of config to be saved.
    :param config: dict. The config to be saved
    :param replace: whether to replace if the config file already existed.
    :return:
    """
    if os.path.isfile(file_path) and not replace:
        with open(file_path, "r") as input_file:
            old_config = json.load(input_file)
        old_config.update(config)
        config = old_config.copy()

    file_dirname = os.path.dirname(os.path.abspath(file_path))
    if file_dirname and not os.path.exists(file_dirname):
        os.makedirs(file_dirname)

    with open(file_path, "w") as output_file:
        json.dump(config, output_file, cls=NumpyEncoder)
