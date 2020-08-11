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
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np

from zoo.orca.data.pandas import read_csv
from zoo.orca.learn.pytorch import Estimator
from zoo.pipeline.api.keras.metrics import Accuracy
from bigdl.optim.optimizer import SGD, EveryEpoch, Adam

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


class TestEstimatorForSpark(TestCase):
    def test_bigdl_pytorch_estimator_shard(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

        model = SimpleModel()

        def loss_func(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        def transform(df):
            result = {
                "x": [df['user'].to_numpy(), df['item'].to_numpy()],
                "y": df['label'].to_numpy()
            }
            return result

        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)

        estimator = Estimator.from_torch(model_creator=model, loss_creator=loss_func,
                                         optimizer_creator=SGD(), backend="bigdl")
        estimator.fit(data=data_shard, epochs=4, batch_size=2, validation_data=data_shard,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())

    def test_bigdl_pytorch_estimator_dataloader(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = self.bn1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        model = SimpleModel()

        estimator = Estimator.from_torch(model_creator=model, loss_creator=nn.BCELoss(),
                                         optimizer_creator=Adam(), backend="bigdl")

        inputs = torch.Tensor([[1, 2], [1, 3], [3, 2], [5, 6], [8, 9], [1, 9]])
        targets = torch.Tensor([[0], [0], [0], [1], [1], [1]])
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
        )
        val_loader = torch.utils.data.DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
        )
        estimator.fit(data=train_loader, epochs=2, validation_data=val_loader,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
