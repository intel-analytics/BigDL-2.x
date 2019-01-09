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


from zoo.pipeline.api.keras.layers import *
from zoo.models.common import ZooModel

from zoo.pipeline.api.keras.engine import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


def createRNN(rnn_type, nlayers, hidden_size):
    _rnn_type = rnn_type.lower()
    if (_rnn_type == "lstm"):
        return [LSTM(hidden_size, return_sequences=True) for layer in range(nlayers)]
    elif (_rnn_type == "gru"):
        return [GRU(hidden_size, return_sequences=True) for layer in range(nlayers)]
    elif (_rnn_type == "simplernn"):
        return [SimpleRNN(hidden_size, return_sequences=True) for layer in range(nlayers)]
    else:
        raise Exception('Only support lstm|gru|simplernn')


class RNNEncoder(ZooKerasLayer):
    """
    A generic recurrent neural network encoder

    # Arguments
    rnns: rnn layers used for encoder, support stacked rnn layers
    embedding: embedding layer in encoder
    input_shape: shape of input, not including batch

    >>> encoder = RNNEncoder.initialize("lstm", 2, 3)
    creating: createZooKerasLSTM
    creating: createZooKerasLSTM
    creating: createZooKerasRNNEncoder

    >>> lstm = LSTM(3)
    creating: createZooKerasLSTM
    >>> embedding = Embedding(1000, 32, input_length=10, name="embedding1")
    creating: createZooKerasEmbedding
    >>> encoder = RNNEncoder([lstm], embedding)
    creating: createZooKerasRNNEncoder
    """
    def __init__(self, rnns, embedding=None, input_shape=None):
        super(RNNEncoder, self).__init__(None,
                                         rnns,
                                         embedding,
                                         list(input_shape) if input_shape else None)

    @classmethod
    def initialize(cls, rnn_type, nlayers, hidden_size, embedding=None, input_shape=None):
        """
        rnn_type: currently support "simplernn | lstm | gru"
        nlayers: number of layers used in encoder
        decoder_hiddenSize: hidden size of encoder
        embedding: embedding layer in encoder, `None` is supported
        """
        rnns = createRNN(rnn_type, nlayers, hidden_size)
        return RNNEncoder(rnns, embedding, input_shape)


class RNNDecoder(ZooKerasLayer):
    """
    A generic recurrent neural network decoder

    # Arguments
    rnns: rnn layers used for decoder, support stacked rnn layers
    embedding: embedding layer in decoder
    input_shape: shape of input, not including batch

    >>> decoder = RNNDecoder.initialize("lstm", 2, 3)
    creating: createZooKerasLSTM
    creating: createZooKerasLSTM
    creating: createZooKerasRNNDecoder

    >>> lstm = LSTM(3)
    creating: createZooKerasLSTM
    >>> embedding = Embedding(1000, 32, input_length=10, name="embedding1")
    creating: createZooKerasEmbedding
    >>> encoder = RNNDecoder([lstm], embedding)
    creating: createZooKerasRNNDecoder
    """
    def __init__(self, rnns, embedding=None, input_shape=None):
        super(RNNDecoder, self).__init__(None,
                                         rnns,
                                         embedding,
                                         list(input_shape) if input_shape else None)

    @classmethod
    def initialize(cls, rnn_type, nlayers, hidden_size, embedding=None, input_shape=None):
        """
        rnn_type: currently support "simplernn | lstm | gru"
        nlayers: number of layers used in decoder
        decoder_hiddenSize: hidden size of decoder
        embedding: embedding layer in decoder, `None` is supported
        """
        rnns = createRNN(rnn_type, nlayers, hidden_size)
        return RNNDecoder(rnns, embedding, input_shape)


class Bridge(ZooKerasLayer):
    """
    defines how to transform encoder to decoder

    # Arguments
    bridge_type: currently only support "dense | densenonlinear"
    decoder_hiddenSize: hidden size of decoder
    bridge: keras layers used to do the transformation

    >>> bridge = Bridge.initialize("dense", 2)
    creating: createZooKerasBridge
    >>> dense = Dense(3)
    creating: createZooKerasDense
    >>> bridge = Bridge.initialize_from_keras_layer(dense)
    creating: createZooKerasBridge
    """
    def __init__(self, bridge_type, decoder_hidden_size, bridge):
        super(Bridge, self).__init__(None, bridge_type, decoder_hidden_size, bridge)

    @classmethod
    def initialize(cls, bridge_type, decoder_hidden_size):
        """
        bridge_type: currently only support "dense | densenonlinear"
        decoder_hiddenSize: hidden size of decoder
        """
        return Bridge(bridge_type, decoder_hidden_size, None)

    @classmethod
    def initialize_from_keras_layer(cls, bridge):
        """
        bridge: keras layers used to do the transformation
        """
        return Bridge("customized", 0, bridge)


class Seq2seq(ZooModel):
    """
    A trainable interface for a simple, generic encoder + decoder model

    # Arguments
    encoder: an encoder object
    decoder: a decoder object
    input_shape: shape of encoder input, for variable length, please use -1 as seq len
    output_shape: shape of decoder input, for variable length, please use -1 as seq len
    bridge: connect encoder and decoder
    generator: Feeding decoder output to generator to generate final result, `None` is supported

    >>> encoder = RNNEncoder.initialize("LSTM", 1, 4)
    creating: createZooKerasLSTM
    creating: createZooKerasRNNEncoder
    >>> decoder = RNNDecoder.initialize("LSTM", 1, 4)
    creating: createZooKerasLSTM
    creating: createZooKerasRNNDecoder
    >>> bridge = Bridge.initialize("dense", 4)
    creating: createZooKerasBridge
    >>> seq2seq = Seq2seq(encoder, decoder, [2, 4], [2, 4], bridge)
    creating: createZooSeq2seq
    """

    def __init__(self, encoder, decoder, input_shape, output_shape, bridge=None,
                 generator=None, bigdl_type="float"):
        super(Seq2seq, self).__init__(None, bigdl_type,
                                      encoder,
                                      decoder,
                                      list(input_shape) if input_shape else None,
                                      list(output_shape) if output_shape else None,
                                      bridge,
                                      generator)
