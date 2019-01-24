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
from zoo.models.seq2seq import *


class TestSeq2seq(ZooTestCase):

    def test_forward_backward(self):
        input_data = [np.random.randint(20, size=(1, 2, 4)),
                      np.random.randint(20, size=(1, 2, 4))]
        encoder = RNNEncoder.initialize("LSTM", 1, 4)
        decoder = RNNDecoder.initialize("LSTM", 1, 4)
        bridge = Bridge.initialize("dense", 4)
        model = Seq2seq(encoder, decoder, [2, 4], [2, 4], bridge)
        self.assert_forward_backward(model, input_data)
        sent1 = np.random.randint(20, size=(1, 2, 4))
        sent2 = np.random.randint(20, size=(1, 4))
        result = model.infer(sent1, sent2, 3)

    def test_save_load(self):
        input_data = [np.random.randint(20, size=(1, 2, 4)),
                      np.random.randint(20, size=(1, 2, 4))]
        encoder = RNNEncoder.initialize("LSTM", 1, 4)
        decoder = RNNDecoder.initialize("LSTM", 1, 4)
        bridge = Bridge.initialize("dense", 4)
        model = Seq2seq(encoder, decoder, [2, 4], [2, 4], bridge)
        self.assert_zoo_model_save_load(model, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
