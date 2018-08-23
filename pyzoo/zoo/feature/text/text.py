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
from bigdl.util.common import JavaValue, callBigDlFunc
from pyspark import RDD

if sys.version >= '3':
    long = int
    unicode = str


class TextFeature(JavaValue):

    def __init__(self, text, label=None, bigdl_type="float"):
        assert isinstance(text, six.string_types), "text of a TextFeature should be a string"
        self.text = text
        self.bigdl_type = bigdl_type
        if label is not None:
            self.label = int(label)
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       text, self.label)
        else:
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       text)

    def get_text(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetText", self.value)

    def get_label(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetLabel", self.value)

    def has_label(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureHasLabel", self.value)

    def keys(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetKeys", self.value)


class TextSet(JavaValue):

    def __init__(self, jvalue, bigdl_type="float"):
        self.value = jvalue
        self.bigdl_type = bigdl_type
        if self.is_local():
            self.text_set = LocalTextSet(jvalue=self.value)
        else:
            self.text_set = DistributedTextSet(jvalue=self.value)

    def is_local(self):
        return callBigDlFunc(self.bigdl_type, "textSetIsLocal", self.value)

    def is_distributed(self):
        return callBigDlFunc(self.bigdl_type, "textSetIsDistributed", self.value)

    def get_word_index(self):
        return callBigDlFunc(self.bigdl_type, "textSetGetWordIndex", self.value)

    def get_texts(self):
        return self.text_set.get_texts()

    def get_labels(self):
        return self.text_set.get_labels()

    def transform(self, transformer, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "transformTextSet", transformer, self.value)
        return self

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        return TextSet(jvalue=callBigDlFunc(bigdl_type, "readTextSet", path, sc, min_partitions))


class LocalTextSet(TextSet):

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        if jvalue:
            self.value = jvalue
        else:
            assert texts, "texts for LocalText can't be None"
            assert all(isinstance(text, six.string_types) for text in texts),\
                "texts should be a list of string"
            if labels is not None:
                labels = map(lambda label: int(label), labels)
                assert all(isinstance(label, int) for label in labels),\
                    "labels should be a list of int"
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       texts, labels)
        self.bigdl_type = bigdl_type

    def get_texts(self):
        return callBigDlFunc(self.bigdl_type, "localTextSetGetTexts", self.value)

    def get_labels(self):
        return callBigDlFunc(self.bigdl_type, "localTextSetGetLabels", self.value)


class DistributedTextSet(TextSet):

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        if jvalue:
            self.value = jvalue
        else:
            assert isinstance(texts, RDD), "texts for DistributedText should be RDD of string"
            if labels is not None:
                assert isinstance(labels, RDD), "labels for DistributedText should be RDD of int"
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       texts, labels.map(lambda x: int(x)))
        self.bigdl_type = bigdl_type

    def get_texts(self):
        return callBigDlFunc(self.bigdl_type, "distributedTextSetGetTexts", self.value)

    def get_labels(self):
        return callBigDlFunc(self.bigdl_type, "distributedTextSetGetLabels", self.value)
