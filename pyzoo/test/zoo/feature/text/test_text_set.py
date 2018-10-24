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

from zoo.feature.common import ChainedPreprocessing
from zoo.feature.text import *
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Embedding, Convolution1D, Flatten, Dense


class TestTextSet:

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_nncontext(init_spark_conf().setMaster("local[1]")
                                 .setAppName("test text set"))
        text1 = "Hello my friend, please annotate my text"
        text2 = "hello world, this is some sentence for my test"
        text3 = "another text for test"
        self.texts = [text1, text2, text3]
        self.labels = [0., 1, 1]
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        self.path = os.path.join(resource_path, "news20")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    @staticmethod
    def _build_model(sequence_length):
        model = Sequential()
        model.add(Embedding(20, 10, input_length=sequence_length))
        model.add(Convolution1D(4, 3))
        model.add(Flatten())
        model.add(Dense(5, activation="softmax"))
        return model

    def test_textset_without_label(self):
        local_set = LocalTextSet(self.texts)
        assert local_set.get_labels() == [-1, -1, -1]
        distributed_set = DistributedTextSet(self.sc.parallelize(self.texts))
        assert distributed_set.get_labels().collect() == [-1, -1, -1]

    def test_textset_convertion(self):
        local_set = LocalTextSet(self.texts, self.labels)
        local1 = local_set.to_local()
        distributed1 = local_set.to_distributed(self.sc)
        assert local1.is_local()
        assert distributed1.is_distributed()
        assert local1.get_texts() == distributed1.get_texts().collect()

        texts_rdd = self.sc.parallelize(self.texts)
        labels_rdd = self.sc.parallelize(self.labels)
        distributed_set = DistributedTextSet(texts_rdd, labels_rdd)
        local2 = distributed_set.to_local()
        distributed2 = distributed_set.to_distributed()
        assert local2.is_local()
        assert distributed2.is_distributed()
        assert local2.get_texts() == distributed2.get_texts().collect()

    def test_local_textset_integration(self):
        local_set = LocalTextSet(self.texts, self.labels)
        assert local_set.is_local()
        assert not local_set.is_distributed()
        assert local_set.get_texts() == self.texts
        assert local_set.get_labels() == self.labels
        tokenized = ChainedPreprocessing([Tokenizer(), Normalizer(), SequenceShaper(10)])(local_set)
        word_index = tokenized.generate_word_index_map(max_words_num=10)
        transformed = ChainedPreprocessing([WordIndexer(word_index),
                                            TextFeatureToSample()])(tokenized)
        assert transformed.is_local()
        word_index = transformed.get_word_index()
        assert len(word_index) == 10
        assert word_index["my"] == 1
        samples = transformed.get_samples()
        assert len(samples) == 3
        for sample in samples:
            assert sample.feature.shape[0] == 10

        model = TestTextSet._build_model(10)
        model.compile("adagrad", "sparse_categorical_crossentropy", ['accuracy'])
        model.fit(transformed, batch_size=2, nb_epoch=2, validation_data=transformed)
        res_set = model.predict(transformed, batch_per_thread=2)
        predicts = res_set.get_predicts()
        for predict in predicts:
            assert len(predict) == 1
            assert predict[0].shape == (5, )
        acc = model.evaluate(transformed, batch_size=2)

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

        tokenized = Tokenizer()(distributed_set)
        transformed = tokenized.normalize().shape_sequence(5)\
            .word2idx().generate_sample()
        word_index = transformed.get_word_index()
        assert len(word_index) == 10
        samples = transformed.get_samples().collect()
        assert len(samples) == 3
        for sample in samples:
            assert sample.feature.shape[0] == 5

        model = TestTextSet._build_model(5)
        model.compile("sgd", "sparse_categorical_crossentropy", metrics=['accuracy'])
        model.fit(transformed, batch_size=2, nb_epoch=2)
        res_set = model.predict(transformed, batch_per_thread=2)
        predicts = res_set.get_predicts().collect()
        for predict in predicts:
            assert len(predict) == 1
            assert predict[0].shape == (5, )
        acc = model.evaluate(transformed, batch_size=2)

    def test_read_local(self):
        local_set = TextSet.read(self.path)
        assert local_set.is_local()
        assert not local_set.get_word_index()  # should be None
        assert len(local_set.get_texts()) == 3
        assert local_set.get_labels() == [0, 0, 1]
        assert local_set.get_samples() == [None, None, None]
        assert local_set.get_predicts() == [None, None, None]

    def test_read_distributed(self):
        distributed_set = TextSet.read(self.path, self.sc, 4)
        assert distributed_set.is_distributed()
        assert not distributed_set.get_word_index()
        assert len(distributed_set.get_texts().collect()) == 3
        assert sorted(distributed_set.get_labels().collect()) == [0, 0, 1]
        assert distributed_set.get_samples().collect() == [None, None, None]
        assert distributed_set.get_predicts().collect() == [None, None, None]


if __name__ == "__main__":
    pytest.main([__file__])
