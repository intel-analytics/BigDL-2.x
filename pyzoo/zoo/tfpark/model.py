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
from bigdl.optim.optimizer import MaxEpoch
from tensorflow.python.keras.engine import training_utils

from zoo.pipeline.api.net import TFDataset, TFOptimizer, TFPredictor
from bigdl.util.common import get_spark_context
import tensorflow.keras.backend as K
import numpy as np
import logging

class Model(object):

    def __init__(self, model=None):
        self.model = model
        self.tf_optimizer = None

    @classmethod
    def from_keras(cls, model):
        return cls(model)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            validation_split=0.,
            validation_data=None,
            distributed=False,
            **kwargs
            ):
        if isinstance(x, TFDataset):
            # todo check arguments
            self._fit_distributed(x, validation_split, epochs, **kwargs)

        elif distributed:
            sc = get_spark_context()
            train_rdd = _create_rdd_x_y(x, y, self.model._feed_input_names, self.model._feed_output_names, sc)

            val_rdd = None
            if validation_data is not None:
                val_rdd = _create_rdd_x_y(validation_data[0], validation_data[1], self.model._feed_input_names, self.model._feed_output_names, sc)

            dataset = TFDataset.from_rdd(train_rdd,
                                         names=self.model._feed_input_names + self.model._feed_output_names,
                                         batch_size=batch_size,
                                         val_rdd=val_rdd)
            self._fit_distributed(dataset, validation_split, epochs, **kwargs)

        else:
            self.model.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           **kwargs
                           )

    def _fit_distributed(self, x, validation_split, epochs, **kwargs):
        if not self.tf_optimizer:
            self.tf_optimizer = TFOptimizer.from_keras(self.model, x, val_spilt=validation_split, **kwargs)
        else:
            self.tf_optimizer.refresh_weights()
        self.tf_optimizer.optimize(MaxEpoch(epochs))

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 distributed=False
                 ):
        if isinstance(x, TFDataset):
            # todo check arguments
            return self._evaluate_distributed(x, batch_size)
        else:
            if distributed:
                sc = get_spark_context()
                rdd = _create_rdd_x_y(x, y, self.model._feed_input_names, self.model._feed_output_names, sc)

                dataset = TFDataset.from_rdd(rdd,
                                             names=self.model._feed_input_names + self.model._feed_output_names,
                                             batch_per_thread=-1 if batch_size is None else batch_size)
                return self._evaluate_distributed(dataset, batch_size)
            else:
                return self.model.evaluate(x=x,
                                           y=y,
                                           batch_size=batch_size)

    def _evaluate_distributed(self, x, batch_size):
        metrics_tensors = [
            self.model.metrics_tensors[m] for m in range(len(self.model.metrics_names)-1)
        ]

        predictor = TFPredictor(K.get_session(), [self.model.total_loss] + metrics_tensors, self.model.inputs + self.model.targets, x)
        result = predictor.predict()
        metrics_sum = result.map(lambda x: x + [np.array(1.0)]).reduce(lambda a,b: elem_sum(a, b))
        length = len(metrics_sum) - 1
        for i in range(length):
            metrics_sum[i] /= metrics_sum[length]
        return metrics_sum[:length]

    def predict(self,
                x,
                batch_size=None,
                distributed=False):

        if isinstance(x, TFDataset):
            # todo check arguments
            return self._predict_distributed(x, batch_size)
        else:
            if distributed:
                sc = get_spark_context()
                rdd = _create_rdd_x(x, self.model._feed_input_names, sc)

                dataset = TFDataset.from_rdd(rdd,
                                             names=self.model._feed_input_names,
                                             batch_per_thread=-1 if batch_size is None else batch_size)
                return np.array(self._predict_distributed(dataset, batch_size).collect())
            else:
                return self.model.predict(x=x,
                                      batch_size=batch_size)

    def _predict_distributed(self, x, batch_size):
        predictor = TFPredictor.from_keras(self.model, x)
        return predictor.predict()

    def train_on_batch(self,
                       x,
                       y=None,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True):
        return self.model.train_on_batch(x=x,
                                         y=y,
                                         sample_weight=sample_weight,
                                         class_weight=class_weight,
                                         reset_metrics=reset_metrics)

    def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
        return self.model.test_on_batch(x=x,
                                        y=y,
                                        sample_weight=sample_weight,
                                        reset_metrics=reset_metrics)

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)


def _create_rdd_x_y(x, y, input_names, output_names, sc):
    x = training_utils.standardize_input_data(x, input_names,
                                              check_batch_axis=False, exception_prefix='input')
    y = training_utils.standardize_input_data(y, output_names,
                                              shapes=None, check_batch_axis=False, exception_prefix='target')

    num_samples = x[0].shape[0]
    num_inputs = len(x)

    input_data = []
    for i in range(num_samples):
        sample = []
        for j in range(num_inputs):
            sample.append(x[j][i])

        for j in range(num_inputs):
            sample.append((y[j][i]))

        input_data.append(sample)

    rdd = sc.parallelize(input_data)
    return rdd


def _create_rdd_x(x, input_names, sc):
    x = training_utils.standardize_input_data(x, input_names,
                                              check_batch_axis=False, exception_prefix='input')

    num_samples = x[0].shape[0]
    num_inputs = len(x)

    input_data = []
    for i in range(num_samples):
        sample = []
        for j in range(num_inputs):
            sample.append(x[j][i])

        input_data.append(sample)

    rdd = sc.parallelize(input_data)
    return rdd


def elem_sum(arr1, arr2):
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
