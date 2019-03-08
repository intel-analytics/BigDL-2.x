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
    def __init__(self, labor, optimizer=None, **kwargs):
        self.labor = labor
        with variable_creator_scope():
            self.labor.build(**kwargs)
            model = self.labor.model
            # Remark: tf.train.optimizers have error when mapped to BigDL optimizer.
            # Two ways to handle this:
            # Either recompile the model using the string or tf.keras.optimizers,
            # or add a mapping between tf.train.optimizers and tf.keras.optimizers.
            if optimizer:
                model.compile(loss=model.loss, optimizer=optimizer, metrics=model.metrics)
            super(TextKerasModel, self).__init__(model)

    # Remark: nlp-architect CRF layer has error when directly loaded by tf.keras.models.load_model.
    # Thus we keep the nlp-architect class as labor and uses its save/load,
    # which only saves the weights with model parameters
    # and reconstruct the model using the exact parameters and setting weights when loading.
    def save_model(self, path):
        self.labor.save(path)

    @staticmethod
    def _load_model(labor, path):
        with variable_creator_scope():
            labor.load(path)
            model = KerasModel(labor.model)
            model.labor = labor
            return model
