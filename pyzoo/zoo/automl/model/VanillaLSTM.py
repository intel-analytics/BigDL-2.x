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

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras
import os

from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class VanillaLSTM(BaseModel):

    def __init__(self, check_optional_config=True, future_seq_len=1):
        """
        Constructor of Vanilla LSTM model
        """
        self.model = None
        self.check_optional_config = check_optional_config
        self.future_seq_len = future_seq_len

    def _build(self, **config):
        """
        build vanilla LSTM model
        :param config: model hyper parameters
        :return: self
        """
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.model = Sequential()
        self.model.add(LSTM(
            # input_shape=(config.get('input_shape_x', 20),
            #             config.get('input_shape_y', 20)),
            units=config.get('lstm_1_units', 20),
            return_sequences=True))
        self.model.add(Dropout(config.get('dropout_1', 0.2)))

        self.model.add(LSTM(
            units=config.get('lstm_2_units', 10),
            return_sequences=False))
        self.model.add(Dropout(config.get('dropout_2', 0.2)))

        self.model.add(Dense(self.future_seq_len))
        self.model.compile(loss='mse',
                           metrics=[self.metric],
                           optimizer=keras.optimizers.RMSprop(lr=config.get('lr', 0.001)))
        return self.model

    def fit_eval(self, x, y, validation_data=None, **config):
        """
        fit for one iteration
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
        dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: 2-d numpy array in format (no. of samples, future sequence length) if future sequence length > 1,
        or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation. If this is specified,
        validation result will be the optimization target for automl. Otherwise, train metric will be the optimization
        target.
        :param config: optimization hyper parameters
        :return: the resulting metric
        """
        # if model is not initialized, __build the model
        if self.model is None:
            self._build(**config)

        hist = self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=config.get('batch_size', 1024),
                              epochs=config.get('epochs', 20),
                              verbose=0
                              )
        # print(hist.history)

        if validation_data is None:
            # get train metrics
            # results = self.model.evaluate(x, y)
            result = hist.history.get(self.metric)[0]
        else:
            result = hist.history.get('val_' + str(self.metric))[0]
        return result

    def evaluate(self, x, y, metric=['mean_squared_error']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        e = Evaluator()
        y_pred = self.predict(x)
        return [e.evaluate(m, y, y_pred) for m in metric]

    def predict(self, x):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        return self.model.predict(x)

    def save(self, model_path, config_path):
        """
        save model to file.
        :param model_path: the model file.
        :param config_path: the config file
        :return:
        """
        self.model.save("vanilla_lstm_tmp.h5")
        os.rename("vanilla_lstm_tmp.h5", model_path)

        config_to_save = {
            "future_seq_len": self.future_seq_len
        }
        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        :return: the restored model
        """
        #self.model = None
        #self._build(**config)
        self.model = keras.models.load_model(model_path)
        #self.model.load_weights(file_path)

    def _get_required_parameters(self):
        return {
            # 'input_shape_x',
            # 'input_shape_y',
            # 'out_units'
        }

    def _get_optional_parameters(self):
        return {
            'lstm_1_units',
            'dropout_1',
            'lstm_2_units',
            'dropout_2',
            'metric',
            'lr',
            'epochs',
            'batch_size'
        }


if __name__ == "__main__":
    model = VanillaLSTM(check_optional_config=False)
    x_train, y_train, x_test, y_test = load_nytaxi_data('../../../../data/nyc_taxi_rolled_split.npz')
    config = {
        # 'input_shape_x': x_train.shape[1],
        # 'input_shape_y': x_train.shape[-1],
        'out_units': 1,
        'dummy1': 1,
        'batch_size': 1024,
        'epochs': 1
    }

    print("fit_eval:",model.fit_eval(x_train, y_train, validation_data=(x_test, y_test), **config))
    print("evaluate:",model.evaluate(x_test, y_test))
    print("saving model")
    model.save("testmodel.tmp.h5",**config)
    print("restoring model")
    model.restore("testmodel.tmp.h5",**config)
    print("evaluate after retoring:",model.evaluate(x_test, y_test))
    os.remove("testmodel.tmp.h5")
