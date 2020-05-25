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

import os.path
import pytest

import numpy as np
import mxnet as mx
import zoo.orca.data.pandas
from zoo.orca.learn.mxnet import MXNetTrainer, create_trainer_config
from test.zoo.orca.learn.mxnet.conftest import get_ray_ctx


def prepare_data(df):
    data = {'input': np.array(df['data'].values.tolist())}
    label = {'label': df['label'].values}
    return {'data': data, 'label': label}


def get_metrics(config):
    return 'accuracy'


def get_model(config):
    input_data = mx.symbol.Variable('input')
    y_true = mx.symbol.Variable('label')
    fc1 = mx.symbol.FullyConnected(data=input_data, num_hidden=20, name='fc1')
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=10, name='fc2')
    output = mx.symbol.SoftmaxOutput(data=fc2, label=y_true, name='output')
    mod = mx.mod.Module(symbol=output,
                        data_names=['input'],
                        label_names=['label'],
                        context=mx.cpu())
    return mod


class TestMXNetXShards(TestCase):
    def test_xshards(self):
        # prepare data
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        self.ray_ctx = get_ray_ctx()
        train_file_path = os.path.join(resource_path, "orca/learn/train_data.json")
        train_data_shard = zoo.orca.data.pandas.read_json(train_file_path, self.ray_ctx,
                                                          orient='records', lines=False)
        train_data_shard.transform_shard(prepare_data)
        test_file_path = os.path.join(resource_path, "orca/learn/test_data.json")
        test_data_shard = zoo.orca.data.pandas.read_json(test_file_path, self.ray_ctx,
                                                         orient='records', lines=False)
        test_data_shard.transform_shard(prepare_data)
        config = create_trainer_config(batch_size=32, log_interval=1, seed=42)
        trainer = MXNetTrainer(config, train_data_shard, get_model, metrics_creator=get_metrics,
                               test_data=test_data_shard)
        trainer.train(nb_epoch=2)


if __name__ == "__main__":
    pytest.main([__file__])
