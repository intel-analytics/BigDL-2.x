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

import warnings


from zoo.pipeline.api.keras.layers import *
from bigdl.util.common import callBigDlFunc
from zoo.models.common.zoo_model import ZooModel

from zoo.pipeline.api.keras.engine import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


class Encoder(ZooKerasLayer):
    """
    The basic encoder class for seqseq model.
    """


class Decoder(ZooKerasLayer):
    """
    The basic Decoder class for seqseq model.
    """


class RNNEncoder(Encoder):
    """
    A generic recurrent neural network encoder

    # Arguments
    rnns: rnn layers used for encoder, support stacked rnn layers
    embedding: embedding layer in encoder
    input_shape: shape of input, not including batch

    >>> encoder = RNNEncoder.initialize("lstm", 2, 3)
    creating: createZooKerasRNNEncoder
    """
    def __init__(self, rnns, embedding, input_shape=None):
        super(RNNEncoder, self).__init__(None,
                                         rnns,
                                         embedding,
                                         list(input_shape) if input_shape else None)

    @classmethod
    def initialize(cls, rnn_type, nlayers, hidden_size, embedding=None, input_shape=None):
        rnns = {
            'lstm': [LSTM(hidden_size, return_sequences=True) for layer in range(nlayers)],
            'gru': [GRU(hidden_size, return_sequences=True) for layer in range(nlayers)],
            'simplernn': [SimpleRNN(hidden_size, return_sequences=True) for layer in range(nlayers)]
        }[rnn_type]()
        RNNEncoder(rnns, embedding, input_shape)


class RNNDecoder(Decoder):
    """
    A generic recurrent neural network decoder

    # Arguments
    rnns: rnn layers used for decoder, support stacked rnn layers
    embedding: embedding layer in decoder
    input_shape: shape of input, not including batch

    >>> decoder = RNNDecoder.initialize("lstm", 2, 3)
    creating: createZooKerasRNNDecoder
    """
    def __init__(self, rnns, embedding, input_shape=None):
        super(RNNDecoder, self).__init__(None,
                                         rnns,
                                         embedding,
                                         list(input_shape) if input_shape else None)

    @classmethod
    def initialize(cls, rnn_type, nlayers, hidden_size, embedding=None, input_shape=None):
        rnns = {
            'lstm': [LSTM(hidden_size, return_sequences=True) for layer in range(nlayers)],
            'gru': [GRU(hidden_size, return_sequences=True) for layer in range(nlayers)],
            'simplernn': [SimpleRNN(hidden_size, return_sequences=True) for layer in range(nlayers)]
        }[rnn_type]()
        RNNDecoder(rnns, embedding, input_shape)


class Bridge(ZooKerasLayer):
    """
    defines how to transform encoder to decoder

    # Arguments
    bridge: keras layers used to do the transformation
    input_shape: shape of input, not including batch

    >>> bridge = Bridge.initialize("dense", 2)
    creating: createZooKerasRNNDecoder
    """
    def __init__(self, bridge_type, decoder_hidden_size, bridge, input_shape=None):
        super(Bridge, self).__init__(None, bridge_type, decoder_hidden_size, bridge,
                                         list(input_shape) if input_shape else None)

    @classmethod
    def initialize(cls, bridge_type, decoder_hidden_size, input_shape=None):
        """
        bridge_type: currently only support "dense | densenonlinear"
        decoder_hiddenSize: hidden size of decoder
        """
        Bridge(bridge_type, decoder_hidden_size, None, input_shape)

    @classmethod
    def initialize_from_layer(cls, bridge, input_shape=None):
        """
        bridge: keras layers used to do the transformation
        """
        Bridge("customized", 0, bridge, input_shape)


class Seq2seq(ZooModel):
    """
    A trainable interface for a simple, generic encoder + decoder model

    # Arguments
    encoder: an encoder object
    decoder: a decoder object
    input_shape: shape of encoder input, for variable length, please input -1
    output_shape: shape of decoder input, for variable length, please input -1
    bridge: connect encoder and decoder
    """

    def __init__(self, encoder, decoder, input_shape, output_shape, bridge=None, generator=None):
        super(Seq2seq, self).__init__(None, encoder, decoder, input_shape, output_shape, bridge, generator)
