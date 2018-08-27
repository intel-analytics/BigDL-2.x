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

from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import *
from zoo.models.common.zoo_model import ZooModel
from bigdl.util.common import callBigDlFunc


if sys.version >= '3':
    long = int
    unicode = str


class TextClassifier(ZooModel):
    """
    The model used for text classification.

    # Arguments
    class_num: The number of text categories to be classified. Positive int.
    embedding_file: The path to the embedding file.
                    Currently only the following GloVe files are supported:
                    "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt"
                    "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
                    You can download them from: https://nlp.stanford.edu/projects/glove/.
    word_index: Dictionary of word (string) and its corresponding index (int).
                The index is supposed to start from 1 with 0 reserved for unknown words.
                During the prediction, if you have words that are not in the word_index
                for the training, you can map them to index 0.
                Default is None. In this case, all the words in the embedding_file will
                be taken into account and you can call
                WordEmbedding.get_word_index(embedding_file) to retrieve the dictionary.
    sequence_length: The length of a sequence. Positive int. Default is 500.
    encoder: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported.
             Default is 'cnn'.
    encoder_output_dim: The output dimension for the encoder. Positive int. Default is 256.
    """
    def __init__(self, class_num, embedding_file, word_index=None, sequence_length=500,
                 encoder="cnn", encoder_output_dim=256, **kwargs):
        if 'token_length' in kwargs:
            kwargs.pop('token_length')
            warnings.warn('The "token_length" argument in TextClassifier has been deprecated '
                          'since 0.3.0, instead you should pass the arguments "embedding_file" '
                          'and "word_index" to construct a TextClassifier with WordEmbedding '
                          'as the first layer.')
        if 'bigdl_type' in kwargs:
            kwargs.pop('bigdl_type')
            self.bigdl_type = kwargs.get("bigdl_type")
        else:
            self.bigdl_type = "float"
        if kwargs:
            raise TypeError('Wrong arguments for TextClassifier: ' + str(kwargs))
        self.class_num = class_num
        self.embedding = WordEmbedding(embedding_file, word_index, input_length=sequence_length)
        self.sequence_length = sequence_length
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        self.model = self.build_model()
        super(TextClassifier, self).__init__(None, self.bigdl_type,
                                             int(class_num),
                                             self.embedding,
                                             int(sequence_length),
                                             encoder,
                                             int(encoder_output_dim),
                                             self.model)

    def build_model(self):
        model = Sequential()
        model.add(self.embedding)
        if self.encoder.lower() == 'cnn':
            model.add(Convolution1D(self.encoder_output_dim, 5, activation='relu'))
            model.add(GlobalMaxPooling1D())
        elif self.encoder.lower() == 'lstm':
            model.add(LSTM(self.encoder_output_dim))
        elif self.encoder.lower() == 'gru':
            model.add(GRU(self.encoder_output_dim))
        else:
            raise ValueError('Unsupported encoder for TextClassifier: ' + self.encoder)
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(self.class_num, activation='softmax'))
        return model

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing TextClassifier model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadTextClassifier", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = TextClassifier
        return model

    def compile(self, optimizer, loss, metrics=None):
        self.model.compile(optimizer, loss, metrics)

    def set_tensorboard(self, log_dir, app_name):
        self.model.set_tensorboard(log_dir, app_name)

    def set_checkpoint(self, path, over_write=True):
        self.model.set_checkpoint(path, over_write)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True):
        self.model.fit(x, y, batch_size, nb_epoch, validation_data, distributed)

    def evaluate(self, x, y=None, batch_size=32):
        return self.model.evaluate(x, y, batch_size)

    def predict(self, x, batch_per_thread=4, distributed=True):
        return self.model.predict(x, batch_per_thread, distributed)
