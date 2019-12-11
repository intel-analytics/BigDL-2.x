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
import tempfile
import os

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.RandomWalk import RandomWalk
from zoo.automl.common.metrics import Evaluator


class TestRandomWalk(ZooTestCase):

    def test_generate_sample(self):
        random_walk = RandomWalk()
        generated_data = random_walk.generate_1d_sample()
        is_random_walk = RandomWalk.random_walk_test(generated_data)
        assert is_random_walk is True

    def test_rolling_fit_eval(self):
        generated_data = RandomWalk.generate_1d_sample(with_white_noise=True)
        train_size = int(len(generated_data) * 0.66)
        train, test = generated_data[0:train_size], generated_data[train_size:]

        random_walk = RandomWalk()
        mse = random_walk.rolling_fit_eval(train_data=train,
                                           test_data=test,
                                           metric=['mse'],
                                           maxiter=1000,
                                           disp=0)
        assert abs(mse[0] - 1.0) < 0.05

    def test_fit_eval(self):
        generated_data = RandomWalk.generate_1d_sample(with_white_noise=True)
        train_size = int(len(generated_data) * 0.66)
        train, test = generated_data[0:train_size], generated_data[train_size:]

        random_walk = RandomWalk()
        mse = random_walk.fit_eval(train_data=train,
                                   test_data=test,
                                   metric=['mse'],
                                   maxiter=1000,
                                   disp=0)
        print(mse[0])

    def test_predict(self):
        generated_data = RandomWalk.generate_1d_sample(with_white_noise=True)
        train_size = int(len(generated_data) * 0.66)
        train, test = generated_data[0:train_size], generated_data[train_size:]

        random_walk = RandomWalk()
        random_walk.rolling_fit_eval(train, test, disp=0)
        result = random_walk.predict(steps=7)

        mse = Evaluator.evaluate('mse', test[0:7], result)
        print(mse)

    def test_save_restore(self):
        generated_data = RandomWalk.generate_1d_sample(with_white_noise=True)
        train_size = int(len(generated_data) * 0.66)
        train, test = generated_data[0:train_size], generated_data[train_size:]

        random_walk = RandomWalk()
        random_walk.rolling_fit_eval(train, test, disp=0)
        result = random_walk.predict(steps=1)

        model_file = tempfile.mktemp(prefix="automl_test_vanilla")
        try:
            random_walk.save(model_file)
            random_walk1 = RandomWalk()
            random_walk1.restore(model_file)
            result1 = random_walk1.predict(steps=1)
            assert result == result1
        finally:
            os.remove(model_file)
