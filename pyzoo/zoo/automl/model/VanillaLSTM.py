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
from zoo.automl.model.base_keras_model import KerasBaseModel
from collections.abc import Iterable


def model_creator(config):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
    import tensorflow as tf

    inp = Input(shape=(None, config["input_dim"]))
    lstm_units = config.get("lstm_units", [32, 32])
    for i, unit in enumerate(lstm_units):
        return_sequences = True if i != len(lstm_units) - 1 else False
        lstm = LSTM(units=unit, return_sequences=return_sequences)(inp)
        inp = Dropout(lstm, rate=config.get("dropouts", [0.2, 0.2]))
    out = Dense(config["output_dim"])(inp)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss="mse",
                  optimizer=getattr(tf.keras.optimizers, config.get("optim", "Adam"))
                                   (learning_rate=config.get("lr", 0.001)),
                  metrics=[config["metric"]])
    return model


def check_iter_type(obj, type):
    return isinstance(obj, type) or all(isinstance(o, type) for o in obj)


class VanillaLSTM(KerasBaseModel):

    def _check_config(self, **config):
        super()._check_config(**config)
        assert isinstance(config["input_dim"], int), "'input_dim' should be int"
        assert isinstance(config["output_dim"], int), "'output_dim' should be int"
        if "lstm_units" in config:
            assert check_iter_type(config["lstm_units"], int), \
                "lstm_units should be int or a list of ints"
        if "dropouts" in config:
            assert check_iter_type(config["dropouts"], float), \
                "dropouts should be float or a list of floats"
        if ("lstm_units" in config and isinstance(config["lstm_units"], Iterable)) and \
                ("dropouts" in config and isinstance(config["dropouts"], Iterable)):
            assert len(config["lstm_units"]) == len(config["dropouts"]), \
                "'lstm_units' and 'dropouts' should have the same length"

    def _get_required_parameters(self):
        return {
            "input_dim",
            "output_dim"
        } | super()._get_required_parameters()

    def _get_optional_parameters(self):
        return {
            "lstm_units",
            "dropouts",
            "optim",
            "lr"
        } | super()._get_optional_parameters()
