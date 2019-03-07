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

from zoo.tfpark import KerasModel, variable_creator_scope


# TODO: add word embedding file support
class TextKerasModel(KerasModel):
    def __init__(self, labor, optimizer=None, *args):
        self.labor = labor
        with variable_creator_scope():
            self.labor.build(*args)
            model = self.labor.model
            # Remark: tf.train.optimizers will cause error when mapped to BigDL optimizer, e.g. lr vs learning_rate.
            # Two ways to handle this: either recompile the model using the corresponding string or tf.keras.optimizer,
            # or add a mapping between tf.train.optimizers and tf.keras.optimizers.
            if optimizer:
                model.compile(loss=model.loss, optimizer=optimizer, metrics=model.metrics)
            super(TextKerasModel, self).__init__(model)

    # Remark: nlp-architect CRF layer has error when directly loaded by tf.keras.models.load_model.
    # Thus we keep the nlp-architect class as labor and uses its save and load method, which only saves the weights
    # and model parameters and reconstruct the model using the exact parameters and setting weights.
    def save_model(self, path):
        self.labor.save(path)

    @staticmethod
    def _load_model(labor, path):
        with variable_creator_scope():
            labor.load(path)
            model = KerasModel(labor.model)
            model.labor = labor
            return model
