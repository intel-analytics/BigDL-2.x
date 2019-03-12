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

from bigdl.nn.layer import Sum

from zoo.pipeline.api.keras.engine import ZooKerasLayer
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential


if sys.version >= '3':
    long = int
    unicode = str


class TransformerLayer(ZooKerasLayer):
    """
    A self attention layer

    # Arguments
    nBlock: block number
    resid_drop: drop probability off projection
    attn_drop: drop probability of attention
    n_head: head number
    mask_attention: whether unidirectional or bidirectional
    embedding_layer: embedding layer
    >>> layer = TransformerLayer.init_with_DefaultEmbedding()
    creating: createZooKerasSequential
    creating: createZooKerasReshape
    creating: createZooKerasEmbedding
    creating: createZooKerasDropout
    creating: createZooKerasReshape
    creating: createSum
    creating: createZooKerasKerasLayerWrapper
    creating: createSqueeze
    creating: createZooKerasKerasLayerWrapper
    creating: createZooKerasTransformerLayer

    """
    def __init__(self, n_block, resid_drop, attn_drop,
                 n_head, mask_attention, embedding_layer):
        super(TransformerLayer, self).__init__(None, n_block, resid_drop, attn_drop, n_head,
                                               mask_attention, embedding_layer)

    @classmethod
    def init_with_default_embedding(cls, vocab=40990, seq_len=77, n_block=12, resid_drop=0.1,
                                    attn_drop=0.1, n_head=12, embedding_size=768,
                                    embedding_drop=0.1, mask_attention=True):
        """
        vocab: vocabulary size of training data, default is 40990
        seq_len: max sequence length of training data, default is 77
        n_block: block number, default is 12
        resid_drop: drop probability of projection, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        n_head: head number, default is 12
        embedding_size: embedding size
        embedding_drop: drop probability of embedding layer, default is 0.1
        mask_attention: whether unidirectional or bidirectional, default is true(unidirectional)
        """
        from bigdl.nn.layer import Squeeze
        embedding = Sequential()

        embedding.add(Reshape([seq_len * 2], input_shape=(seq_len, 2)))\
            .add(Embedding(vocab, embedding_size, input_length=seq_len * 2))\
            .add(Dropout(embedding_drop))\
            .add(Reshape((seq_len, 2, embedding_size)))\
            .add(KerasLayerWrapper(Sum(dimension=3, squeeze=True)))
        # walk around for bug #1208, need remove this line after the bug fixed
        embedding.add(KerasLayerWrapper(Squeeze(dim=3)))

        return TransformerLayer(n_block, resid_drop, attn_drop, n_head, mask_attention, embedding)
