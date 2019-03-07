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

import nlp_architect.models.intent_extraction as intent_models
from zoo.tfpark.text import TextKerasModel


class IntentEntity(TextKerasModel):
    def __init__(self, num_intents, num_entities, word_length, word_vocab_size,
                 char_vocab_size, word_emb_dim=100, char_emb_dim=30,
                 char_lstm_dim=30, tagger_lstm_dim=100, dropout=0.2, optimizer='adam'):
        super(IntentEntity, self).__init__(intent_models.MultiTaskIntentModel(use_cudnn=False),
                                           optimizer, word_length, num_entities, num_intents,
                                           word_vocab_size, char_vocab_size, word_emb_dim,
                                           char_emb_dim, char_lstm_dim, tagger_lstm_dim, dropout)

    @staticmethod
    def load_model(path):
        labor = intent_models.MultiTaskIntentModel(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = IntentEntity
        return model
