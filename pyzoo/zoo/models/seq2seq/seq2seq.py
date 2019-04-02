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

from bigdl.nn.layer import Layer
from zoo.pipeline.api.keras.layers import *
from zoo.models.common import ZooModel

from zoo.pipeline.api.keras.engine import ZooKerasLayer
from zoo.pipeline.api.keras.models import Model


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
        hidden_size: hidden size of encoder
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
        hidden_size: hidden size of decoder
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
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKerasSelectTable
    creating: createZooKerasModel
    creating: createZooSeq2seq
    """

    def __init__(self, encoder, decoder, input_shape, output_shape, bridge=None,
                 generator=None, bigdl_type="float"):
        if (input_shape is None) or (output_shape is None):
            raise TypeError('input_shape and output_shape cannot be None')
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        self.bridge = bridge
        self.generator = generator
        self.bigdl_type = bigdl_type
        self.model = self.build_model()
        super(Seq2seq, self).__init__(None, self.bigdl_type,
                                      self.encoder,
                                      self.decoder,
                                      self.input_shape,
                                      self.output_shape,
                                      self.bridge,
                                      self.generator,
                                      self.model)

    def build_model(self):
        encoder_input = Input(name="encoder_input", shape=self.input_shape)
        decoder_input = Input(name="decoder_input", shape=self.output_shape)
        encoder_output = self.encoder(encoder_input)

        encoder_final_states = SelectTable(1)(encoder_output)
        decoder_init_states =\
            self.bridge(encoder_final_states) if self.bridge else encoder_final_states

        decoder_output = self.decoder([decoder_input, decoder_init_states])

        output = self.generator(decoder_output) if self.generator else decoder_output

        return Model([encoder_input, decoder_input], output)

    def set_checkpoint(self, path, over_write=True):
        callBigDlFunc(self.bigdl_type, "seq2seqSetCheckpoint",
                      self.value,
                      path,
                      over_write)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing Seq2seq model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadSeq2seq", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = Seq2seq
        return model

    # For the following methods, please refer to KerasNet for documentation.
    def compile(self, optimizer, loss, metrics=None):
        if isinstance(optimizer, six.string_types):
            optimizer = to_bigdl_optim_method(optimizer)
        if isinstance(loss, six.string_types):
            loss = to_bigdl_criterion(loss)
        if metrics and all(isinstance(metric, six.string_types) for metric in metrics):
            metrics = to_bigdl_metrics(metrics, loss)
        callBigDlFunc(self.bigdl_type, "seq2seqCompile",
                      self.value,
                      optimizer,
                      loss,
                      metrics)

    def fit(self, x, batch_size=32, nb_epoch=10, validation_data=None):
        callBigDlFunc(self.bigdl_type, "seq2seqFit",
                      self.value,
                      x,
                      batch_size,
                      nb_epoch,
                      validation_data)

    def infer(self, input, start_sign, max_seq_len=30, stop_sign=None, build_output=None):
        """
        Inference API for given input

        # Arguments
        input: a sequence of data feed into encoder, eg: batch x seqLen x featureSize
        start_sign: a ndarray which represents start and is fed into decoder
        max_seq_len: max sequence length for final output
        stop_sign: a ndarray that indicates model should stop infer further if current output
        is the same with stopSign
        build_output: Feeding model output to buildOutput to generate final result
        """
        jinput, input_is_table = Layer.check_input(input)
        assert not input_is_table
        jstart_sign, start_sign_is_table = Layer.check_input(start_sign)
        assert not start_sign_is_table
        if stop_sign:
            jstop_sign, stop_sign_is_table = Layer.check_input(stop_sign)
            assert not start_sign_is_table
        else:
            jstop_sign = None
        results = callBigDlFunc(self.bigdl_type, "seq2seqInfer",
                                self.value,
                                jinput[0],
                                jstart_sign[0],
                                max_seq_len,
                                jstop_sign[0] if jstop_sign else None,
                                build_output)
        return results
