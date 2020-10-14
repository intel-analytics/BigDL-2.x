# Copyright 2018 Analytics Zoo Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ray.tune.utils import flatten_dict
import numpy as np

VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32]
VALID_NP_TYPES = [np.float32, np.float64, np.int32]


class TensorboardXLogger():
    def __init__(self, logs_dir=None, writer=None):
        self.logs_dir = logs_dir
        self._file_writer = None
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print("pip install tensorboardx to see TensorBoard log files.")
            raise
        if writer:
            self._file_writer = writer
        else:
            self._file_writer = SummaryWriter(logdir=self.logs_dir)
    
    def run(self, config, metric):
        assert len(config.keys() - metric.keys()) == 0
        
        # validation check
        new_metric = {}
        for key in metric.keys():
            new_metric[key] = {}
            for k, value in metric[key].items():
                if (type(value) in VALID_SUMMARY_TYPES):
                    if type(value) in VALID_NP_TYPES:
                        if np.isnan(value):
                            continue
                    new_metric[key][k] = value
        new_config = {}
        for key in config.keys():
            new_config[key] = {}
            for k, value in config[key].items():
                if type(value) is not None:
                    new_config[key][k] = value
        
        # hparams log write
        for key in new_metric.keys():
            # new_config[key]["address"] = key
            self._file_writer.add_hparams(new_config[key], new_metric[key])

    def close(self):
        self._file_writer.close()
        
