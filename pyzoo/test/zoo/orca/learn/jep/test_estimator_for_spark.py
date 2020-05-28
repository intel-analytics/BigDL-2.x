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
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo.common.nncontext import *
from zoo.pipeline.api.keras.metrics import Accuracy
from bigdl.optim.optimizer import EveryEpoch


class TestEstimatorForSpark(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_estimator_graph_fit(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

        from bigdl.optim.optimizer import SGD
        import zoo.orca.data.pandas
        from zoo.orca.learn.pytorch.estimator import Estimator

        model = SimpleModel()

        def lossFunc(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)

        def transform(df):
            result = {
                "x": [df['user'].to_numpy(), df['item'].to_numpy()],
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.spark_trainer(model, lossFunc, SGD())
        est.fit(data_shard=data_shard,
                batch_size=8,
                steps=10,
                validation_data_shard=data_shard,
                validation_methods=[Accuracy()],
                checkpoint_trigger=EveryEpoch())


if __name__ == "__main__":
    pytest.main([__file__])
