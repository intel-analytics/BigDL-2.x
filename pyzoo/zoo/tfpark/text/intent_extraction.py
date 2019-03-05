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

from zoo.tfpark.text import TextKerasModel
import nlp_architect.models.intent_extraction as intent_models


class IntentAndEntity(TextKerasModel):
    def __init__(self, word_length, num_labels, num_intent_labels, word_vocab_size,
                 char_vocab_size, word_emb_dims=100, char_emb_dims=30,
                 char_lstm_dims=30, tagger_lstm_dims=100, dropout=0.2, optimizer='adam'):
        super(IntentAndEntity, self).__init__(intent_models.MultiTaskIntentModel(use_cudnn=False), optimizer,
                                              word_length, num_labels, num_intent_labels,
                                              word_vocab_size, char_vocab_size, word_emb_dims,
                                              char_emb_dims, char_lstm_dims, tagger_lstm_dims, dropout)

    @staticmethod
    def load_model(path):
        labor = intent_models.MultiTaskIntentModel(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = IntentAndEntity
        return model
