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

from keras.layers import *
from keras.models import Model
from zoo.models.textranker import KNRM
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestKNRM(ZooTestCase):

    # Model definition from MatchZoo rewritten in Keras 1.2.2
    def keras_knrm(self, text1_length, text2_length, vocab_size, embed_size,
                   kernel_num=21, sigma=0.1, exact_sigma=0.001):
        def Kernel_layer(mu,sigma):
            def kernel(x):
                return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return Activation(kernel)

        query = Input(name='query', shape=(text1_length,))
        doc = Input(name='doc', shape=(text2_length,))
        embedding = Embedding(vocab_size, embed_size, name="embedding")
        q_embed = embedding(query)
        d_embed = embedding(doc)
        mm = merge([q_embed, d_embed], mode="dot", dot_axes=[2, 2])

        KM = []
        for i in range(kernel_num):
            mu = 1. / (kernel_num - 1) + (2. * i) / (kernel_num - 1) - 1.0
            sigma = sigma
            if mu > 1.0:
                sigma = exact_sigma
                mu = 1.0
            mm_exp = Kernel_layer(mu, sigma)(mm)
            mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x, 2))(mm_exp)
            mm_log = Activation(K.tf.log1p)(mm_doc_sum)
            mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log)
            KM.append(mm_sum)

        Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)
        out_ = Dense(1, init="uniform", activation="sigmoid", name="dense")(Phi)

        model = Model([query, doc], out_)
        return model

    def test_with_keras(self):
        kmodel = self.keras_knrm(5, 10, 20, 100)
        input_data = np.random.randint(20, size=(4, 15))
        koutput = kmodel.predict([input_data[:, :5], input_data[:, 5:]])
        kweights = kmodel.get_weights()
        bweights = [kweights[0], np.transpose(kweights[1]), kweights[2]]
        model = KNRM(5, 10, 20, 100)  # Model definition on Scala side
        model.set_weights(bweights)
        output = model.forward(input_data)
        self.assert_allclose(output, koutput)

    def test_save_load(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
