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
from zoo.models.recommendation import UserItemFeature
from zoo.models.recommendation import NeuralCF
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestNeuralCF(ZooTestCase):

    def test_forward_backward_without_mf(self):
        model = NeuralCF(30, 30, 2, include_mf=False)
        input_data = np.random.randint(1, 30,  size=(10, 2))
        self.assert_forward_backward(model, input_data)

    def test_forward_backward_with_mf(self):
        model = NeuralCF(10, 10, 5, 5, 5)
        input_data = np.random.randint(1, 10, size=(3, 2))
        self.assert_forward_backward(model, input_data)

    def test_save_load(self):
        model = NeuralCF(10000, 2000, 10)
        input_data = np.random.randint(100, 2000, size=(300, 2))
        self.assert_zoo_model_save_load(model, input_data)

    def test_predict_recommend(self):
        def gen_rand_user_item_feature(user_num, item_num, class_num):
            user_id = random.randint(1, user_num)
            item_id = random.randint(1, item_num)
            rating = random.randint(1, class_num)
            sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
            return UserItemFeature(user_id, item_id, sample)
        model = NeuralCF(200, 80, 5)
        data = self.sc.parallelize(range(0, 50))\
            .map(lambda i: gen_rand_user_item_feature(200, 80, 5))
        predictions = model.predict_user_item_pair(data).collect()
        print(predictions[0])
        recommended_items = model.recommend_for_user(data, max_items=3).collect()
        print(recommended_items[0])
        recommended_users = model.recommend_for_item(data, max_users=4).collect()
        print(recommended_users[0])

    def test_compile_fit(self):
        def gen_rand_user_item_feature(user_num, item_num, class_num):
            user_id = random.randint(1, user_num)
            item_id = random.randint(1, item_num)
            rating = random.randint(1, class_num)
            sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
            return UserItemFeature(user_id, item_id, sample)
        model = NeuralCF(200, 80, 5)
        model.summary()
        data = self.sc.parallelize(range(0, 50)) \
            .map(lambda i: gen_rand_user_item_feature(200, 80, 5)) \
            .map(lambda pair: pair.sample)
        model.compile(optimizer="adam",
                      loss=SparseCategoricalCrossEntropy(zero_based_label=False),
                      metrics=['accuracy'])
        tmp_log_dir = create_tmp_path()
        model.set_tensorboard(tmp_log_dir, "training_test")
        model.fit(data, nb_epoch=1, batch_size=32, validation_data=data)
        train_loss = model.get_train_summary("Loss")
        val_loss = model.get_validation_summary("Loss")
        print(np.array(train_loss))
        print(np.array(val_loss))


if __name__ == "__main__":
    pytest.main([__file__])
