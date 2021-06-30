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

from zoo.chronos.data import TSDataset
from zoo.automl.common.metrics import Evaluator

class TSPipeline:
    def __init__(self, best_model, best_config, **kwargs):
        self._best_model = best_model
        self._best_config = best_config
        if "scaler" in kwargs.keys():
            self._scaler = kwargs["scaler"]
            self._scaler_index = kwargs["scaler_index"]

    def evaluate(self, data, metrics=['mse'], multioutput="raw_values", batch_size=32):
        '''
        Evaluate the time series pipeline.

        :param data: data can be a TSDataset or data creator(will be supported).
               the TSDataset should follow the same operations as the training
               TSDataset used in AutoTSTrainer.fit.
        :param metrics:
        :param multioutput:
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        x, y = self._tsdataset_to_numpy(data, is_predict=False)
        yhat = self.predict(x, batch_size=batch_size)
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat, multioutput=multioutput)
                       for m in metrics]
        return eval_result

    def predict(self, data, batch_size=32):
        '''
        Rolling predict with time series pipeline.

        :param data: data can be a TSDataset or data creator(will be supported).
               the TSDataset should follow the same operations as the training
               TSDataset used in AutoTSTrainer.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        x, _ = self._tsdataset_to_numpy(data, is_predict=True)
        yhat = self._best_model.predict(x, batch_size=batch_size)
        if self._scaler:
            from zoo.chronos.data.utils import unscale_timeseries_numpy
            yhat = unscale_timeseries_numpy(y, self._scaler, self._scaler_index)
        return yhat

    def fit(self, data, validation_data=None, epochs=1, metric="mse"):
        '''
        Incremental fitting

        :param data: data can be a TSDataset or data creator(will be supported).
               the TSDataset should follow the same operations as the training
               TSDataset used in AutoTSTrainer.fit.
        :param validation_data: validation data, same format as data.
        :param epochs: incremental fitting epoch. The value defaults to 1.
        :param metric: evaluate metric.
        '''
        x, y = self._tsdataset_to_numpy(data, is_predict=False)
        if validation_data is None:
            x_val, y_val = self._tsdataset_to_numpy(validation_data, is_predict=False)
        else:
            x_val, y_val = x, y

        res = self._best_model.fit_eval(data=(x, y), validation_data=(x_val, y_val), metric=metric)
        return res

    def save(self, file_path):
        # if not os.path.isdir(file_path):
        #     os.mkdir(file_path)
        # config_path = os.path.join(file_path, "config.json")
        # model_path = os.path.join(file_path, "weights_tune.h5")
        # if feature_transformers is not None:
        #     feature_transformers.save(config_path, replace=True)
        # if model is not None:
        #     model.save(model_path, config_path)
        # if config is not None:
        #     save_config(config_path, config)
        pass

    @staticmethod
    def load(self, file_path):
        pass

    def _tsdataset_to_numpy(self, data, is_predict=False):
        if isinstance(data, TSDataset):
            lookback = self._best_config["past_seq_len"]
            horizon = 0 if is_predict else self._best_config["future_seq_len"]
            selected_features = self._best_config["selected_feature"]
            data.roll(lookback, horizon, feature_col=selected_features)
            x, y = data.to_numpy()
        else:
            raise NotImplementedError("Data creator has not been supported now.")
        return x, y
