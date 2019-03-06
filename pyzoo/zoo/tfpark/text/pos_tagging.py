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

from nlp_architect.models import chunker
from tensorflow.python.keras.models import Model
from zoo.tfpark import KerasModel
from zoo.tfpark.text import TextKerasModel


class SequenceTagger(TextKerasModel):
    def __init__(self, num_pos_labels, num_chunk_labels, word_vocab_size,
                 char_vocab_size=None, word_length=25, feature_size=100, dropout=0.2,
                 classifier='softmax', optimizer='adam'):
        super(SequenceTagger, self).__init__(chunker.SequenceTagger(use_cudnn=False), None,
                                             word_vocab_size, num_pos_labels, num_chunk_labels, char_vocab_size,
                                             word_length, feature_size, dropout, classifier, optimizer)

    @staticmethod
    def load_model(path):
        labor = chunker.SequenceTagger(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = SequenceTagger
        return model


class POSTagger(KerasModel):
    """
    The model uses only the pos output of IntentAndEntity.
    """
    def __init__(self, num_pos_labels, word_vocab_size, char_vocab_size=None,
                 word_length=25, feature_size=100, dropout=0.2, optimizer='adam'):
        labor = SequenceTagger(num_pos_labels, 2, word_vocab_size, char_vocab_size,
                               word_length, feature_size, dropout, optimizer=optimizer)
        model = Model(labor.model.input, labor.model.output[0])
        model.compile(loss=labor.model.loss["pos_output"], optimizer=optimizer,
                      metrics=[labor.model.metrics["pos_output"]])
        super(POSTagger, self).__init__(model)

    @staticmethod
    def load_model(path):
        model = KerasModel.load_model(path)
        model.__class__ = POSTagger
        return model

# TODO: SequenceChunker will have load error if it extends KerasModel as it may contain CRF
