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

from zoo.pipeline.api.net import TFDataset, TFOptimizer, TFPredictor
import tensorflow.keras.backend as K

class Model(object):

    def __init__(self, model=None):
        self.model = model
        self.tf_optimizer = None


    @classmethod
    def from_keras(cls, model):
        return cls(model)

    def get_weights(self):
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs
            ):
        if isinstance(x, TFDataset):
            # todo check arguments
            self._fit_distributed(x, validation_split, **kwargs)
        else:
            self.model.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=callbacks,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           class_weight=class_weight,
                           sample_weight=sample_weight,
                           initial_epoch=initial_epoch,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           max_queue_size=max_queue_size,
                           workers=workers,
                           use_multiprocessing=use_multiprocessing,
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
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False
                 ):
        if isinstance(x, TFDataset):
            # todo check arguments
            self._evaluate_distributed(x, batch_size)
        else:
            self.model.evaluate(x=x,
                                y=y,
                                batch_size=batch_size,
                                verbose=verbose,
                                sample_weight=sample_weight,
                                steps=steps,
                                callbacks=callbacks,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                use_multiprocessing=use_multiprocessing)

    def _evaluate_distributed(self, x, batch_size):
        metrics_tensors = [
            self.model._all_stateful_metrics_tensors[m] for m in self.model.metrics_names[1:]
        ]
        predictor = TFPredictor.from_outputs(K.get_session(), [self.model.total_loss] + metrics_tensors)
        result = predictor.predict()
        metrics_sum = result.map(lambda x: x.append(1.0)).reduce(elem_sum)
        length = len(metrics_sum) - 1
        for i in range(length):
            metrics_sum[i] /= metrics_sum[length]
        return metrics_sum[:length]


    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):

        if isinstance(x, TFDataset):
            # todo check arguments
            self._predict_distributed(x, batch_size)
        else:
            self.model.predict(x=x,
                               batch_size=batch_size,
                               verbose=verbose,
                               steps=steps,
                               callbacks=callbacks,
                               max_queue_size=max_queue_size,
                               workers=workers,
                               use_multiprocessing=use_multiprocessing)

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


def elem_sum(arr1, arr2):
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result