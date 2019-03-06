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

import tensorflow as tf
import nlp_architect.models.ner_crf as ner_model
from nlp_architect.models.intent_extraction import Seq2SeqIntentModel
from zoo.tfpark.text import TextKerasModel


class NERCRF(TextKerasModel):
    def __init__(self, num_entities, word_vocab_size, word_length, char_vocab_size,
                 word_emb_dim=100, char_emb_dim=30, tagger_lstm_dim=100, dropout=0.5,
                 crf_mode='reg', optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=5.)):
        super(NERCRF, self).__init__(ner_model.NERCRF(use_cudnn=False), optimizer,
                                     word_length, num_entities, word_vocab_size, char_vocab_size, word_emb_dim,
                                     char_emb_dim, 20, tagger_lstm_dim, dropout, crf_mode)
        # Remark: In nlp-architect NERCRF.build(..), word_lstm_dims is never used. Thus, remove this argument here.

    @staticmethod
    def load_model(path):
        labor = ner_model.NERCRF(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = NERCRF
        return model
