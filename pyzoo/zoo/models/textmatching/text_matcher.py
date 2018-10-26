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

from zoo.models.common.zoo_model import ZooModel

if sys.version >= '3':
    long = int
    unicode = str


class TextMatcher(ZooModel):
    """
    The base class for text matching models in Analytics Zoo.
    Referred to MatchZoo implementation: https://github.com/NTMC-Community/MatchZoo
    """
    def __init__(self, text1_length, vocab_size, embed_size=300, embed_weights=None,
                 train_embed=True, bigdl_type="float"):
        self.text1_length = text1_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed_weights = embed_weights
        self.train_embed = train_embed
        self.bigdl_type = bigdl_type
