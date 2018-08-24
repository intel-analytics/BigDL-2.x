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

from zoo.feature.text import *
from zoo.common.nncontext import *


class TestTextSet:

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_nncontext("test textset")
        self.text1 = "Hello my friend, please annotate my text"
        self.text2 = "hello world, this is some sentence for my test"
        self.text3 = "dummy text for test"
        self.texts = [self.text1, self.text2, self.text3]
        self.labels = [0., 1, 1]
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        self.path = os.path.join(resource_path, "news20")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_text_feature_with_label(self):
        feature1 = TextFeature(self.text1, 1)
        assert feature1.get_text() == self.text1
        assert feature1.get_label() == 1
        assert feature1.has_label()
        assert set(feature1.keys()) == {'text', 'label'}

    def test_text_feature_without_label(self):
        feature = TextFeature(self.text1)
        assert feature.get_text() == self.text1
        assert feature.get_label() == -1
        assert not feature.has_label()
        assert feature.keys() == ['text']

    def test_local_textset(self):
        local_set = LocalTextSet(self.texts, self.labels)
        assert local_set.is_local()
        assert not local_set.is_distributed()
        assert local_set.get_texts() == self.texts
        assert local_set.get_labels() == self.labels

    def test_distributed_textset_integration(self):
        texts_rdd = self.sc.parallelize(self.texts)
        labels_rdd = self.sc.parallelize(self.labels)
        distributed_set = DistributedTextSet(texts_rdd, labels_rdd)
        assert distributed_set.is_distributed()
        assert not distributed_set.is_local()
        assert distributed_set.get_texts().collect() == self.texts
        assert distributed_set.get_labels().collect() == self.labels

        sets = distributed_set.random_split([0.5, 0.5])
        train_texts = sets[0].get_texts().collect()
        test_texts = sets[1].get_texts().collect()
        assert set(train_texts + test_texts) == set(self.texts)
        train_labels = sets[0].get_labels().collect()
        test_labels = sets[1].get_labels().collect()
        assert set(train_labels + test_labels) == set(self.labels)

        transformed = distributed_set.tokenize().normalize().word2idx().shape_sequence(5).gen_sample()
        word_index = transformed.get_word_index()
        assert word_index["my"] == 1
        assert word_index.has_key("hello")
        assert not word_index.has_key("Hello")
        assert len(word_index) == 14

    def test_read_local(self):
        local_set = TextSet.read(self.path)
        assert local_set.is_local()
        assert not local_set.get_word_index()  # should be None

    def test_read_distributed(self):
        distributed_set = TextSet.read(self.path, self.sc, 4)
        assert distributed_set.is_distributed()
        assert not distributed_set.get_word_index()


if __name__ == "__main__":
    pytest.main([__file__])
