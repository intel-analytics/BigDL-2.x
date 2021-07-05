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
from zoo.models.common import ZooModel
from zoo.models.textmatching import TextMatcher
from zoo.pipeline.api.keras.layers import Input, Embedding, Dense, Squeeze, prepare_embedding
from zoo.pipeline.api.keras.models import Model
from bigdl.util.common import JTensor
from zoo.common.utils import callZooFunc

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
    embedding_file: The path to the word embedding file.
                    Currently only the following GloVe files are supported:
                    "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt"
                    "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
                    You can download from: https://nlp.stanford.edu/projects/glove/.
    word_index: Dictionary of word (string) and its corresponding index (int).
                The index is supposed to start from 1 with 0 reserved for unknown words.
                During the prediction, if you have words that are not in the word_index
                for the training, you can map them to index 0.
                Default is None. In this case, all the words in the embedding_file will
                be taken into account and you can call
                WordEmbedding.get_word_index(embedding_file) to retrieve the dictionary.
    train_embed: Boolean. Whether to train the embedding layer or not. Default is True.
    kernel_num: Int > 1. The number of kernels to use. Default is 21.
    sigma: Float. Defines the kernel width, or the range of its softTF count. Default is 0.1.
    exact_sigma: Float. The sigma used for the kernel that harvests exact matches
                 in the case where RBF mu=1.0. Default is 0.001.
    target_mode: String. The target mode of the model. Either 'ranking' or 'classification'.
                 For ranking, the output will be the relevance score between text1 and text2 and
                 you are recommended to use 'rank_hinge' as loss for pairwise training.
                 For classification, the last layer will be sigmoid and the output will be the
                 probability between 0 and 1 indicating whether text1 is related to text2 and
                 you are recommended to use 'binary_crossentropy' as loss for binary classification.
                 Default mode is 'ranking'.
    """

    def __init__(self, text1_length, text2_length, embedding_file, word_index=None,
                 train_embed=True, kernel_num=21, sigma=0.1, exact_sigma=0.001,
                 target_mode="ranking", bigdl_type="float"):
        embed_weights = prepare_embedding(embedding_file, word_index,
                                          randomize_unknown=True, normalize=True)
        vocab_size, embed_size = embed_weights.shape
        super(KNRM, self).__init__(text1_length, vocab_size, embed_size,
                                   embed_weights, train_embed, target_mode, bigdl_type)
        self.text2_length = text2_length
        assert kernel_num > 1, "kernel_num must be an int larger than 1"
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
                                          self.target_mode,
                                          self.model)

    def build_model(self):
        # Remark: Share weights for embedding is not supported.
        # Thus here the model takes concatenated input and slice to split the input.
        input = Input(name='input', shape=(self.text1_length + self.text2_length,))
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
            mm_exp = A.exp((-0.5) * (mm - mu) * (mm - mu) / sigma / sigma)
            mm_doc_sum = A.sum(mm_exp, axis=2)
            mm_log = A.log(mm_doc_sum + 1.0)
            # Remark: Keep the reduced dimension for the last sum and squeeze after stack.
            # Otherwise, when batch=1, the output will become a Scalar not compatible for stack.
            mm_sum = A.sum(mm_log, axis=1, keepDims=True)
            KM.append(mm_sum)
        Phi = Squeeze(2)(A.stack(KM))
        if self.target_mode == "ranking":
            output = Dense(1, init="uniform")(Phi)
        else:
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
        jmodel = callZooFunc(bigdl_type, "loadKNRM", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = KNRM
        return model
