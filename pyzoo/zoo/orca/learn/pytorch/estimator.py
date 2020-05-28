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
from bigdl.optim.optimizer import MaxIteration, SGD
from zoo.pipeline.estimator.estimator import Estimator as SparkEstimator
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.feature.common import FeatureSet


class Estimator(object):
    def __init__(self, model, loss, optimizer, model_dir=None, bigdl_type="float"):
        self.loss = loss
        self.model = model
        self.estimator = SparkEstimator(model, optimizer, model_dir, bigdl_type=bigdl_type)

    def fit(self, data_shard, steps, batch_size, validation_data_shard=None,
            validation_methods=None, checkpoint_trigger=None):
        from zoo.orca.data.utils import to_sample
        end_trigger = MaxIteration(steps)

        assert batch_size > 0, \
            "batch_size should be greater than 0"

        train_rdd = data_shard.rdd.flatMap(to_sample)
        train_feature_set = FeatureSet.sample_rdd(train_rdd)
        val_feature_set = None if validation_data_shard is None \
            else FeatureSet.sample_rdd(validation_data_shard.rdd.flatMap(to_sample))

        self.estimator.train(train_feature_set, self.loss, end_trigger, checkpoint_trigger,
                             val_feature_set, validation_methods, batch_size)
        return self

    @staticmethod
    def spark_trainer(model, loss=None, optimizer=None, model_dir=None):
        torch_model = TorchModel.from_pytorch(model)
        if loss is None:
            torch_loss = TorchLoss()
        else:
            torch_loss = TorchLoss.from_pytorch(loss)
        if optimizer is None:
            optimizer = SGD()

        return Estimator(torch_model, torch_loss, optimizer, model_dir)
