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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import tensorflow.keras as keras
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
        self.feature_num = None
        self.metric = None
        self.batch_size = None

    def _get_dropout(self, input_tensor, p=0.5, mc=False):
        if mc:
            return Dropout(p)(input_tensor, training=True)
        else:
            return Dropout(p)(input_tensor)

    def _build(self, mc=False, **config):
        """
        build vanilla LSTM model
        :param config: model hyper parameters
        :return: self
        """
        super()._check_config(**config)
        self.metric = config.get('metric', 'mean_squared_error')
        self.batch_size = config.get('batch_size', 1024)

        inp = Input(shape=(None, self.feature_num))
        lstm_1 = LSTM(units=config.get('lstm_1_units', 20),
                      return_sequences=True)(inp)
        dropout_1 = self._get_dropout(lstm_1,
                                      p=config.get('dropout_1', 0.2),
                                      mc=mc)
        lstm_2 = LSTM(units=config.get('lstm_2_units', 10),
                      return_sequences=False)(dropout_1)
        dropout_2 = self._get_dropout(lstm_2,
                                      p=config.get('dropout_2', 0.2),
                                      mc=mc)
        out = Dense(self.future_seq_len)(dropout_2)
        self.model = Model(inputs=inp, outputs=out)

        # self.model = Sequential()
        # self.model.add(LSTM(
        #     # input_shape=(config.get('input_shape_x', 20),
        #     #             config.get('input_shape_y', 20)),
        #     units=config.get('lstm_1_units', 20),
        #     return_sequences=True))
        # self.model.add(Dropout(config.get('dropout_1', 0.2)))
        #
        # self.model.add(LSTM(
        #     units=config.get('lstm_2_units', 10),
        #     return_sequences=False))
        # self.model.add(Dropout(config.get('dropout_2', 0.2)))

        # self.model.add(Dense(self.future_seq_len))
        self.model.compile(loss='mse',
                           metrics=[self.metric],
                           optimizer=keras.optimizers.RMSprop(lr=config.get('lr', 0.001)))
        return self.model

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, **config):
        """
        fit for one iteration
        :param x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
        in the last dimension, the 1st col is the time index (data type needs to be numpy datetime
        type, e.g. "datetime64"),
        the 2nd col is the target value (data type should be numeric)
        :param y: 2-d numpy array in format (no. of samples, future sequence length)
        if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
        if future sequence length = 1
        :param validation_data: tuple in format (x_test,y_test), data used for validation.
        If this is specified, validation result will be the optimization target for automl.
        Otherwise, train metric will be the optimization target.
        :param config: optimization hyper parameters
        :return: the resulting metric
        """
        self.feature_num = x.shape[2]
        # if model is not initialized, __build the model
        if self.model is None:
            self._build(mc=mc, **config)

        hist = self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=self.batch_size,
                              epochs=config.get('epochs', 20),
                              verbose=verbose
                              )
        # print(hist.history)

        if validation_data is None:
            # get train metrics
            # results = self.model.evaluate(x, y)
            result = hist.history.get(self.metric)[0]
        else:
            result = hist.history.get('val_' + str(self.metric))[0]
        return result

    def evaluate(self, x, y, metric=['mse']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x)
        return [Evaluator.evaluate(m, y, y_pred) for m in metric]

    def predict(self, x, mc=False):
        """
        Prediction on x.
        :param x: input
        :return: predicted y
        """
        return self.model.predict(x)

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.future_seq_len))

        for i in range(n_iter):
            result[i, :, :] = self.predict(x)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def save(self, model_path, config_path):
        """
        save model to file.
        :param model_path: the model file.
        :param config_path: the config file
        :return:
        """
        self.model.save(model_path)
        # os.rename("vanilla_lstm_tmp.h5", model_path)

        config_to_save = {
            # "future_seq_len": self.future_seq_len,
            "metric": self.metric,
            "batch_size": self.batch_size
        }
        save_config(config_path, config_to_save)

    def restore(self, model_path, **config):
        """
        restore model from file
        :param model_path: the model file
        :param config: the trial config
        :return: the restored model
        """
        # self.model = None
        # self._build(**config)
        self.model = keras.models.load_model(model_path)
        # self.model.load_weights(file_path)

        # self.future_seq_len = config["future_seq_len"]
        # for continuous training
        self.metric = config["metric"]
        self.batch_size = config["batch_size"]

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
