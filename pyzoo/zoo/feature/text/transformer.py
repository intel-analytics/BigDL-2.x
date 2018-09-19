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
    """
    Base class of Transformers that transform TextFeature.
    """
    def __init__(self, bigdl_type="float", *args):
        super(TextTransformer, self).__init__(bigdl_type, *args)


class Tokenizer(TextTransformer):
    """
    Transform text to array of string tokens.

    >>> tokenizer = Tokenizer()
    creating: createTokenizer
    """
    def __init__(self, bigdl_type="float"):
        super(Tokenizer, self).__init__(bigdl_type)


class Normalizer(TextTransformer):
    """
    Removes all dirty characters from tokens and convert to lower case.
    Original tokens will be replaced by normalized tokens.

    >>> normalizer = Normalizer()
    creating: createNormalizer
    """
    def __init__(self, bigdl_type="float"):
        super(Normalizer, self).__init__(bigdl_type)


class WordIndexer(TextTransformer):
    """
    Given a wordIndex map, transform tokens to corresponding indices.
    Those words not in the map will be aborted.

    # Arguments
    map: Dict with word as its key and index as its value.

    >>> word_indexer = WordIndexer(map={"it": 1, "me": 2})
    creating: createWordIndexer
    """
    def __init__(self, map, bigdl_type="float"):
        super(WordIndexer, self).__init__(bigdl_type, map)


class SequenceShaper(TextTransformer):
    """
    Shape the sequence of indexedTokens or tokens to a fixed length.
    If the original sequence is longer than the target length, it will be truncated from
    the beginning or the end.
    If the original sequence is shorter than the target length, it will be padded to the end.

    # Arguments
    len: The target length.
    trunc_mode: Truncation mode. Either 'pre' or 'post'. Default is 'pre'.
                If 'pre', the sequence will be truncated from the beginning.
                If 'post', the sequence will be truncated from the end.
    input_key: The key for the sequence. Either 'tokens' or 'indexedTokens'.
               The original sequence will be replaced by the shaped sequence.
               Default is 'indexedTokens'.
    pad_element: The element to be padded to the sequence if the original length is
                 smaller than the target length.
                 It should be a string if input_key is 'tokens'.
                 It should be an integer if input_key is 'indexedTokens'.
                 Default is 0 for 'indexedTokens' with the convention that we reserve index
                 0 for unknown words.
    >>> sequence_shaper = SequenceShaper(len=6, trunc_mode="post")
    creating: createSequenceShaper

    >>> sequence_shaper2 = SequenceShaper(6, "pre", "tokens", "dummy")
    creating: createSequenceShaper
    """
    def __init__(self, len, trunc_mode="pre", input_key="indexedTokens",
                 pad_element=0, bigdl_type="float"):
        assert isinstance(pad_element, int) or isinstance(pad_element, six.string_types), \
            "pad_element should be int or string"
        super(SequenceShaper, self).__init__(bigdl_type, len, trunc_mode,
                                             input_key, pad_element)


class TextFeatureToSample(TextTransformer):
    """
    Transform indexedTokens and label (if any) of a TextFeature to a BigDL Sample.

    >>> to_sample = TextFeatureToSample()
    creating: createTextFeatureToSample
    """
    def __init__(self, bigdl_type="float"):
        super(TextFeatureToSample, self).__init__(bigdl_type)
