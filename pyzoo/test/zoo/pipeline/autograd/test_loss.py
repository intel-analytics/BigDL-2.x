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

import keras.backend as KK
import keras.layers as klayers
import keras.objectives as kloss
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *


np.random.seed(1337)  # for reproducibility


class TestLoss(ZooTestCase):
    # kkloss is a function which accept y_pred and y_true.
    # y_pred and y_true are all keras_tensor
    # zloss is a AbstractCriterion
    # The first dim of shape is batch
    def compareLossWithKeras(self, kkloss_func, zloss, shape, sizeAverageKerasLoss=True):

        y_pred = klayers.Input(shape=shape[1:])
        y_true = klayers.Input(shape=shape[1:])

        batch = shape[0]

        kkloss = kkloss_func(y_true, y_pred)
        y_true_value = np.random.uniform(0, 1, shape)
        y_pred_value = np.random.uniform(0, 1, shape)

        k_grad_y_pred = KK.get_session().run(
            KK.gradients(kkloss, y_pred),
            feed_dict={y_true: y_true_value, y_pred: y_pred_value})[0]
        k_output = KK.get_session().run(kkloss,
                                        feed_dict={y_true: y_true_value, y_pred: y_pred_value})
        z_output = zloss.forward(y_true_value, y_pred_value)
        z_grad_y_pred = zloss.backward(y_true_value, y_pred_value)

        assert(z_output == pytest.approx(np.mean(k_output), 1e-5, 1e-5))
        if sizeAverageKerasLoss:
            self.assert_allclose(k_grad_y_pred / batch, z_grad_y_pred)
        else:
            self.assert_allclose(k_grad_y_pred, z_grad_y_pred)

    def test_abs(self):
        def mean_absolute_error(y_true, y_pred):
            result = mean(abs(y_true - y_pred), axis=1)
            return result
        self.compareLossWithKeras(kloss.mean_absolute_error,
                                  CustomLoss(mean_absolute_error, [3]), [2, 3],
                                  sizeAverageKerasLoss=True)

    def test_rank_hinge_loss(self):
        def rank_hinge_loss(**kwargs):
            if isinstance(kwargs, dict) and 'batch' in kwargs:
                batch = kwargs['batch']

            def _rank_hinge_loss(y_true, y_pred):
                y_pred = y_pred + y_true - y_true
                margin = 1.0
                pos = merge([y_pred.slice(0, i, 1) for i in range(0, batch, 2)],
                            mode="concat", concat_axis=0)
                neg = merge([y_pred.slice(0, i, 1) for i in range(1, batch, 2)],
                            mode="concat", concat_axis=0)
                loss = maximum(margin + neg - pos, 0.)
                return loss
            return _rank_hinge_loss

        def keras_rank_hinge_loss(y_true, y_pred):
            margin = 1.0
            y_pos = klayers.Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
            y_neg = klayers.Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
            loss = KK.maximum(0., margin + y_neg - y_pos)
            return KK.mean(loss)

        batch = 32
        self.compareLossWithKeras(keras_rank_hinge_loss,
                                  CustomLoss(rank_hinge_loss(batch=batch), [1]),
                                  [batch, 1], sizeAverageKerasLoss=False)

    def test_abs_with_fit(self):
        def mean_absolute_error(y_true, y_pred):
            result = mean(abs(y_true - y_pred), axis=1)
            return result
        data_len = 1000
        X_ = np.random.uniform(0, 1, (1000, 2))
        Y_ = ((2 * X_).sum(1) + 0.4).reshape([data_len, 1])
        model = Sequential()
        model.add(Dense(1, input_shape=(2, )))
        model.compile(optimizer=SGD(learningrate=1e-2),
                      loss=mean_absolute_error,
                      metrics=["auc"])
        model.fit(x=X_,
                  y=Y_,
                  batch_size=32,
                  nb_epoch=500,
                  validation_data=None,
                  distributed=False)
        w = model.get_weights()
        self.assert_allclose(w[0], np.array([2, 2]).reshape([1, 2]), rtol=1e-1)
        self.assert_allclose(w[1], np.array([0.4]), rtol=1e-1)
        predict_result = model.predict_local(X_)
        self.assert_allclose(Y_, predict_result.reshape((data_len, 1)), rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
