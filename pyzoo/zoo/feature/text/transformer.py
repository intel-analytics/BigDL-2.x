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
import six
from zoo.feature.common import Preprocessing

if sys.version >= '3':
    long = int
    unicode = str


class TextTransformer(Preprocessing):

    def __init__(self, bigdl_type="float", *args):
        super(TextTransformer, self).__init__(bigdl_type, *args)


class Tokenizer(TextTransformer):
    """
    >>> tokenizer = Tokenizer()
    creating: createTokenizer
    """
    def __init__(self, out_key="tokens", bigdl_type="float"):
        super(Tokenizer, self).__init__(bigdl_type, out_key)


class Normalizer(TextTransformer):
    """
    >>> normalizer = Normalizer()
    creating: createNormalizer
    """
    def __init__(self, out_key="tokens", bigdl_type="float"):
        super(Normalizer, self).__init__(bigdl_type, out_key)


class WordIndexer(TextTransformer):
    """
    >>> word_indexer = WordIndexer(map={"it": 1, "me": 2})
    creating: createWordIndexer
    """
    def __init__(self, map, bigdl_type="float"):
        super(WordIndexer, self).__init__(bigdl_type, map)


class SequenceShaper(TextTransformer):
    """
    >>> sequence_shaper = SequenceShaper(len=6, mode="post")
    creating: createSequenceShaper

    >>> sequence_shaper2 = SequenceShaper(6, "pre", "tokens", "dummy")
    creating: createSequenceShaper
    """
    def __init__(self, len, mode="pre", input_key="indexedTokens",
                 pad_element=0, bigdl_type="float"):
        assert isinstance(pad_element, int) or isinstance(pad_element, six.string_types), \
            "pad_element should be int or string"
        super(SequenceShaper, self).__init__(bigdl_type, len, mode,
                                             input_key, pad_element)


class TextFeatureToSample(TextTransformer):
    """
    >>> to_sample = TextFeatureToSample()
    creating: createTextFeatureToSample
    """
    def __init__(self, bigdl_type="float"):
        super(TextFeatureToSample, self).__init__(bigdl_type)
