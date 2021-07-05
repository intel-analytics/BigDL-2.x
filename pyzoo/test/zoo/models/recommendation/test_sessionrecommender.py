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

import pytest

import random

from zoo.pipeline.api.keras.layers import *
from zoo.models.recommendation import SessionRecommender
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestSessionRecommender(ZooTestCase):
    def test_forward_backward_without_history(self):
        model = SessionRecommender(30, 5, [10, 5], 2)
        input_data = np.random.randint(1, 30, size=(10, 2))
        self.assert_forward_backward(model, input_data)

    def test_forward_backward_with_history(self):
        model = SessionRecommender(30, 5, [10, 5], 2, True, [6, 3], 5)
        input_data = [np.random.randint(1, 30, size=(10, 2)),
                      np.random.randint(1, 30, size=(10, 5))]
        self.assert_forward_backward(model, input_data)

    def test_save_load(self):
        model = SessionRecommender(30, 5, [10, 5], 2, True, [6, 3], 5)
        input_data = [np.random.randint(1, 30, size=(10, 2)),
                      np.random.randint(1, 30, size=(10, 5))]
        self.assert_zoo_model_save_load(model, input_data)

    def test_compile_fit(self):
        model = SessionRecommender(30, 5, [10, 5], 2, True, [6, 3], 5)
        input_data = [[np.random.randint(1, 30, size=(2)),
                       np.random.randint(1, 30, size=(5)),
                       np.random.randint(1, 30)] for i in range(100)]
        samples = self.sc.parallelize(input_data)\
            .map(lambda x: Sample.from_ndarray((x[0], x[1]), np.array(x[2])))
        train, test = samples.randomSplit([0.8, 0.2], seed=1)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['top5Accuracy'])
        model.fit(train, batch_size=4, nb_epoch=1, validation_data=test)

    def test_recommed_predict(self):
        model = SessionRecommender(30, 5, [10, 5], 2, True, [6, 3], 5)
        input_data = [[np.random.randint(1, 30, size=(2)),
                       np.random.randint(1, 30, size=(5)),
                       np.random.randint(1, 30)] for i in range(100)]
        samples = [Sample.from_ndarray((input_data[i][0], input_data[i][1]),
                                       np.array(input_data[i][2])) for i in range(100)]
        rdd = self.sc.parallelize(samples)
        results1 = model.predict(rdd).collect()
        print(results1[0])

        recommendations1 = model.recommend_for_session(rdd, 3, zero_based_label=False).collect()
        print(recommendations1[0])

        recommendations2 = model.recommend_for_session(samples, 3, zero_based_label=False)
        print(recommendations2[0])


if __name__ == "__main__":
    pytest.main([__file__])
