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

import pytest
import os
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.feature.text import *

text1 = "Hello my friend, please annotate my text"
text2 = "hello world, this is some sentence for my test"
texts = [text1, text2]
labels = [0., 1]
resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
news20_path = os.path.join(resource_path, "news20")


class TestTextSet(ZooTestCase):
    def test_text_feature_with_label(self):
        feature1 = TextFeature(text1, 0.)
        feature2 = TextFeature(text2, 1)
        assert feature1.get_text() == text1
        assert feature1.get_label() == 0
        assert feature1.has_label()
        assert feature2.get_text() == text2
        assert feature2.get_label() == 1
        assert feature2.has_label()
        assert feature1.keys() == ['text', 'label']

    def test_text_feature_without_label(self):
        feature = TextFeature(text1)
        assert feature.get_text() == text1
        assert feature.get_label() == -1
        assert not feature.has_label()
        assert feature.keys() == ['text']

    def test_local_textset(self):
        local_set = LocalTextSet(texts, labels)
        assert local_set.is_local()
        assert not local_set.is_distributed()
        assert local_set.get_texts() == texts
        assert local_set.get_labels() == labels

    def test_distributed_textset(self):
        texts_rdd = self.sc.parallelize(texts)
        labels_rdd = self.sc.parallelize(labels)
        distributed_set = DistributedTextSet(texts_rdd, labels_rdd)
        assert distributed_set.is_distributed()
        assert not distributed_set.is_local()
        assert distributed_set.get_texts().collect() == texts
        assert distributed_set.get_labels().collect() == labels

    def test_read_local(self):
        local_set = TextSet.read(news20_path)
        assert local_set.is_local()
        assert not local_set.get_word_index()  # should be None

    def test_read_distributed(self):
        distributed_set = TextSet.read(news20_path, self.sc, 4)
        assert distributed_set.is_distributed()
        assert not distributed_set.get_word_index()


if __name__ == "__main__":
    pytest.main([__file__])
