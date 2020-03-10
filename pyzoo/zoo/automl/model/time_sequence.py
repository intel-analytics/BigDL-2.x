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
from .abstract import BaseModel
from .VanillaLSTM import VanillaLSTM
from .Seq2Seq import LSTMSeq2Seq


class TimeSequenceModel(BaseModel):
    """
    Time Sequence Model is used to do model selection.
    """
    def __init__(self, check_optional_config=False, future_seq_len=None):
        """
        Contructor of time sequence model
        :param check_optional_config:
        :param future_seq_len:
        """
        if future_seq_len:
            self._model_selection(future_seq_len, check_optional_config)

    def _model_selection(self, future_seq_len, check_optional_config=False, verbose=1):
        if future_seq_len == 1:
            self.model = VanillaLSTM(check_optional_config=check_optional_config,
                                     future_seq_len=future_seq_len)
            if verbose == 1:
                print("Model selection: Vanilla LSTM model is selected.")
        else:
            self.model = LSTMSeq2Seq(check_optional_config=check_optional_config,
                                     future_seq_len=future_seq_len)
            if verbose == 1:
                print("Model selection: LSTM Seq2Seq model is selected.")

    def fit_eval(self, x, y, validation_data=None, verbose=0, **config):
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
        return self.model.fit_eval(x, y, validation_data=validation_data, verbose=verbose, **config)

    def evaluate(self, x, y, metric=['mean_squared_error']):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        return self.model.evaluate(x, y, metric)

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
        self.model.save(model_path, config_path)

    def restore(self, model_path, **config):
        self._model_selection(future_seq_len=config["future_seq_len"], verbose=0)
        self.model.restore(model_path, **config)

    def _get_required_parameters(self):
        return self.model._get_required_parameters()

    def _get_optional_parameters(self):
        return self.model._get_optional_parameters()





