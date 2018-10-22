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
from zoo.pipeline.api.keras.layers import Input, Embedding, Dense, Squeeze
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
        # Remark: Share weights for embedding is not supported.
        # Thus here the model takes concatenated input and slice to split the input.
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
            mm_exp = A.exp(-0.5 * (mm - mu) * (mm - mu) / sigma / sigma)
            mm_doc_sum = A.sum(mm_exp, 2)
            mm_log = A.log(mm_doc_sum + 1.0)
            # Remark: Keep the reduced dimension for the last sum and squeeze after stack.
            # Otherwise, when batch=1, the output will become a Scalar not compatible for stack.
            mm_sum = A.sum(mm_log, 1, keepDims=True)
            KM.append(mm_sum)
        Phi = Squeeze(2)(A.stack(KM, 1))
        output = Dense(1, init="uniform", activation="sigmoid")(Phi)
        model = Model(input=input, output=output)
        return model
