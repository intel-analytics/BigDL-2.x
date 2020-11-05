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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.logger import TensorboardXLogger
import numpy as np
import random
import os.path


class TestTensorboardXLogger(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_tbxlogger_valid_type(self):
        trail_num = 100
        test_config = {}
        test_metric = {}
        for i in range(trail_num):
            test_config["run_{}".format(i)] =\
                {"config_good": random.randint(8, 96),
                 "config_unstable": None if random.random() < 0.5 else 1,
                 "config_bad": None}
            test_metric["run_{}".format(i)] =\
                {"matrix_good": random.randint(0, 100)/100,
                 "matrix_unstable": np.nan if random.random() < 0.5 else 1,
                 "matrix_bad": np.nan}
        logger = TensorboardXLogger(os.path.abspath(os.path.expanduser("~/test_tbxlogger")))
        logger.run(test_config, test_metric)
        logger.close()

    def test_tbxlogger_keys(self):
        test_config = {"run1": {"lr": 0.01}}
        test_metric = {"run2": {"lr": 0.02}}
        logger = TensorboardXLogger(os.path.abspath(os.path.expanduser("~/test_tbxlogger")))
        with pytest.raises(Exception):
            logger.run(test_config, test_metric)
        logger.close()
