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

import six
from bigdl.util.common import JavaValue, callBigDlFunc
from pyspark import RDD


class TextFeature(JavaValue):

    def __init__(self, text, label=None, bigdl_type="float"):
        self.text = text
        self.bigdl_type = bigdl_type
        if label != None:
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


class LocalTextSet(TextSet):

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        if jvalue:
            self.value = jvalue
        else:
            assert texts, "texts for LocalText can't be None"
            assert all(isinstance(text, six.string_types) for text in texts),\
                "texts should be a list of string"
            if labels != None:
                labels = map(lambda label: int(label), labels)
                assert all(isinstance(label, int) for label in labels),\
                    "labels should be a list of int"
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       texts, labels)
        self.bigdl_type = bigdl_type


class DistributedTextSet(TextSet):

    def __init__(self, texts=None, labels=None, jvalue=None, bigdl_type="float"):
        if jvalue:
            self.value = jvalue
        else:
            assert isinstance(texts, RDD), "texts for DistributedText should be RDD of string"
            if labels != None:
                assert isinstance(labels, RDD), "labels for DistributedText should be RDD of int"
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       texts, labels)
        self.bigdl_type = bigdl_type
