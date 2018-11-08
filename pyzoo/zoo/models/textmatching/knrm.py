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

import zoo.pipeline.api.autograd as A
from zoo.models.common.zoo_model import ZooModel
from zoo.models.textmatching.text_matcher import TextMatcher
from zoo.pipeline.api.keras.layers import Input, Embedding, Dense, Squeeze
from zoo.pipeline.api.keras.models import Model
from bigdl.util.common import callBigDlFunc, JTensor

if sys.version >= '3':
    long = int
    unicode = str


class KNRM(TextMatcher):
    """
    Kernel-pooling Neural Ranking Model with RBF kernel.
    https://arxiv.org/abs/1706.06613

    # Arguments:
    text1_length: Sequence length of text1 (query).
    text2_length: Sequence length of text2 (doc).
    vocab_size: Int. The input_dim of the embedding layer. Ought to be the total number
                of words in the corpus +1, with index 0 reserved for unknown words.
    embed_size: Int. The output_dim of the embedding layer. Default is 300.
    embed_weights: Numpy array. Pre-trained word embedding weights if any. Default is None
                   and in this case, initial weights will be randomized.
    train_embed: Boolean. Whether to train the embedding layer or not. Default is True.
    kernel_num: Int. The number of kernels to use. Default is 21.
    sigma: Float. Defines the kernel width, or the range of its softTF count. Default is 0.1.
    exact_sigma: Float. The sigma used for the kernel that harvests exact matches
                 in the case where RBF mu=1.0. Default is 0.001.
    """
    def __init__(self, text1_length, text2_length, vocab_size, embed_size=300, embed_weights=None,
                 train_embed=True, kernel_num=21, sigma=0.1, exact_sigma=0.001, bigdl_type="float"):
        super(KNRM, self).__init__(text1_length, vocab_size, embed_size,
                                   embed_weights, train_embed, bigdl_type)
        self.text2_length = text2_length
        self.kernel_num = kernel_num
        self.sigma = float(sigma)
        self.exact_sigma = float(exact_sigma)
        self.model = self.build_model()
        super(TextMatcher, self).__init__(None, self.bigdl_type,
                                          self.text1_length,
                                          self.text2_length,
                                          self.vocab_size,
                                          self.embed_size,
                                          JTensor.from_ndarray(embed_weights),
                                          self.train_embed,
                                          self.kernel_num,
                                          self.sigma,
                                          self.exact_sigma,
                                          self.model)

    def build_model(self):
        # Remark: Share weights for embedding is not supported.
        # Thus here the model takes concatenated input and slice to split the input.
        input = Input(name='input', shape=(self.text1_length + self.text2_length, ))
        embedding = Embedding(self.vocab_size, self.embed_size,
                              weights=self.embed_weights, trainable=self.train_embed)(input)
        query_embed = embedding.slice(1, 0, self.text1_length)
        doc_embed = embedding.slice(1, self.text1_length, self.text2_length)
        mm = A.batch_dot(query_embed, doc_embed, axes=[2, 2])  # Translation Matrix.
        KM = []
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:  # Exact match.
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

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing KNRM model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadKNRM", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = KNRM
        return model
