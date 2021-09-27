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
from unittest import TestCase

import numpy as np
import pytest

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from zoo.ray import RayContext
from zoo.orca.learn.mxnet import Estimator, create_config

np.random.seed(1337)  # for reproducibility


def get_train_data_iter(config, kv):
    train_data = np.random.rand(200, 30)
    train_label = np.random.randint(0, 10, (200,))
    train = mx.io.NDArrayIter(train_data, train_label,
                              batch_size=config["batch_size"], shuffle=True)
    return train


def get_test_data_iter(config, kv):
    test_data = np.random.rand(80, 30)
    test_label = np.random.randint(0, 10, (80,))
    test = mx.io.NDArrayIter(test_data, test_label,
                             batch_size=config["batch_size"], shuffle=True)
    return test


def get_model(config):
    class SimpleModel(gluon.Block):
        def __init__(self, **kwargs):
            super(SimpleModel, self).__init__(**kwargs)
            self.fc1 = nn.Dense(20)
            self.fc2 = nn.Dense(10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    net = SimpleModel()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.cpu()])
    return net


def get_loss(config):
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_metrics(config):
    return mx.metric.Accuracy()


class TestMXNetGluon(TestCase):
    def test_gluon(self):
        current_ray_ctx = RayContext.get()
        address_info = current_ray_ctx.address_info
        assert "object_store_address" in address_info
        config = create_config(log_interval=2, optimizer="adam",
                               optimizer_params={'learning_rate': 0.02})
        estimator = Estimator(config, get_model, get_loss,
                              eval_metrics_creator=get_metrics,
                              validation_metrics_creator=get_metrics,
                              num_workers=2)
        estimator.fit(get_train_data_iter, validation_data=get_test_data_iter, epochs=2)
        estimator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
