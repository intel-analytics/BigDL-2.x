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

import numpy as np
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.tfpark.text import *


class TestTextModel(ZooTestCase):

    def test_intent_entity(self):
        model = IntentEntity(num_intents=8, num_entities=5, word_length=10,
                             word_vocab_size=200, char_vocab_size=50)
        input_data = [np.random.randint(200, size=(8, 30)), np.random.randint(50, size=(8, 30, 10))]
        output = model.predict(input_data, distributed=True)
        assert isinstance(output, list) and len(output) == 2
        assert output[0].shape == (8, 8)
        assert output[1].shape == (8, 30, 5)
        self.assert_tfpark_model_save_load(model, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
