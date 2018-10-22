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

import sys

from zoo.models.common.zoo_model import ZooModel
import zoo.pipeline.api.autograd as A
from zoo.pipeline.api.keras.layers import Input, Embedding, Flatten, Dense
from zoo.pipeline.api.keras.models import Model

if sys.version >= '3':
    long = int
    unicode = str


class KNRM(ZooModel):
    """

    """
    def __init__(self, text1_length, text2_length, vocab_size, embed_size, embed_weights=None,
                 kernel_num=21, sigma=0.1, exact_sigma=0.001, bigdl_type="float"):
        self.text1_length = text1_length
        self.text2_length = text2_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed_weights = embed_weights
        self.kernel_num = kernel_num
        self.sigma = float(sigma)
        self.exact_sigma = float(exact_sigma)
        super(KNRM, self).__init__(None, bigdl_type,
                                   text1_length,
                                   text2_length,
                                   vocab_size,
                                   embed_size,
                                   embed_weights,
                                   kernel_num,
                                   sigma,
                                   exact_sigma)

    def build_model(self):
        def RBF(mu, sigma):
            def kernel(x):
                return A.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return A.Lambda(lambda x: kernel(x))

        input = Input(name='input', shape=(self.text1_length + self.text2_length, ))
        embedding = Embedding(self.vocab_size, self.embed_size, weights=self.embed_weights)(input)
        query_embed = embedding.slice(1, 0, self.text1_length)
        doc_embed = embedding.slice(1, self.text1_length, self.text2_length)
        mm = A.batch_dot(query_embed, doc_embed, axes=[2, 2])

        KM = []
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            mm_exp = RBF(mu, sigma)(mm)
            mm_doc_sum = A.Lambda(lambda x: A.sum(x, 2))(mm_exp)
            mm_log = A.Lambda(lambda x: A.log(x + 1.0))(mm_doc_sum)
            # Keep the reduced dimension for the last sum.
            # Otherwise, when batch=1, the output will become a Scalar not compatible for the stack operation.
            mm_sum = A.Lambda(lambda x: A.sum(x, 1, keepDims=True))(mm_log)
            KM.append(mm_sum)

        KMStack = A.stack(KM, 1)
        flatten = Flatten()(KMStack)  # Remove the extra dimension from keepDims of the last sum.
        output = Dense(1, init="uniform", activation="sigmoid")(flatten)

        model = Model(input=input, output=output)
        return model
